import re
import sys
import requests
import json
import sqlite3
import pdfplumber
import tracker
from html.parser import HTMLParser
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from your .env file
load_dotenv(Path(__file__).resolve().parent / ".env")

# Initialize the Gemini model for CrewAI
# (Make sure GEMINI_API_KEY is in your .env file)
gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite"
)

# ==========================================
# 1. WEB FETCHING TOOL
# ==========================================

class _TextExtractor(HTMLParser):
    """Strips HTML down to plain readable text."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav", "aside"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self):
        text = "".join(self.parts)
        # Collapse whitespace and blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def fetch_webpage_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return f"The URL did not return HTML (Content-Type: {content_type})."

    extractor = _TextExtractor()
    extractor.feed(response.text)
    text = extractor.get_text()

    # Keep output bounded so it fits in the model's context
    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated ...]"
    return text


# ==========================================
# 2. DEFINE AGENTS (Using Gemini LLM)
# ==========================================

job_researcher = Agent(
    role='Job Market Research Analyst',
    goal='Analyze job postings fetched from real URLs. Extract tasks, key skills, qualifications, and core keywords required for the role and produce an accurate summary.',
    backstory=(
        "You are an expert technical recruiter and career coach with years of experience "
        "analyzing job postings to understand what hiring managers and ATS (Applicant Tracking Systems) look for."
    ),
    verbose=True,
    llm=gemini_llm
)

profile_analyst = Agent(
    role='Candidate Profile Analyst',
    goal='Read the candidate\'s CV and certificates and produce a structured, accurate summary of their skills, experience, education, and certifications.',
    backstory=(
        "You are an expert career coach who carefully reviews resumes, transcripts, and certificates "
        "to build a precise picture of a candidate's real qualifications — without inventing or exaggerating anything."
    ),
    verbose=True,
    llm=gemini_llm
)

fit_analyst = Agent(
    role='Candidate-Job Fit Analyst',
    goal='Compare a job listing against a candidate profile and produce an honest fit assessment: a rating, matched skills, and skill gaps.',
    backstory=(
        "You are a pragmatic technical recruiter who evaluates candidate-job fit with calibrated honesty — "
        "you highlight genuine matches, flag real gaps, and never inflate a candidate's suitability."
    ),
    verbose=True,
    llm=gemini_llm
)

application_specialist = Agent(
    role='Application & Positioning Specialist',
    goal=(
        "Identify candidate-job matches and transferable skills and translate them into compelling resume content "
        "and honest gap-bridging arguments."
    ),
    backstory=(
        "You are an experienced career strategist and application writer who excels at framing a "
        "candidate's real background to meet a target role's technical demands. You highlight genuine "
        "strengths, reframe transferable skills to bridge gaps, and never invent qualifications."
    ),
    verbose=True,
    llm=gemini_llm
)

cover_letter_writer = Agent(
    role='Cover Letter Writer (German)',
    goal='Write a tailored, professional cover letter in German that stays under 2340 characters and is structured in the three blocks "Sie", "Ich", and "Wir".',
    backstory=(
        "You are an experienced German business writer who crafts concise, authentic cover letters "
        "(Anschreiben) that connect a candidate's real qualifications to the specific role — "
        "never exaggerating and never using empty phrases."
    ),
    verbose=True,
    llm=gemini_llm
)


# ==========================================
# 3. INPUT DATA
# ==========================================

# Paste a real job posting URL here, or pass it as a command-line argument:
#   python application_pipeline.py https://example.com/jobs/123
job_url = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "https://example.com/jobs/123"
)

job_text = fetch_webpage_text(job_url)


def extract_pdf_texts(max_chars_per_pdf: int | None = None) -> str:
    """Extracts the full text from every PDF in the same folder as this script."""
    script_dir = Path(__file__).resolve().parent
    pdf_files = sorted(script_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit("No PDF files found in the script folder.")

    sections = []
    for pdf_path in pdf_files:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if max_chars_per_pdf is not None and len(text) > max_chars_per_pdf:
            text = text[:max_chars_per_pdf] + "\n[... truncated ...]"
        sections.append(f"===== {pdf_path.name} =====\n{text}")

    return "\n\n".join(sections)


candidate_documents = extract_pdf_texts()


# Optional cover letter draft used as inspiration for the cover letter agent.
# Place a file named cover_letter_draft.txt in this folder to enable it.
def load_cover_letter_draft() -> str:
    script_dir = Path(__file__).resolve().parent
    for filename in ("cover_letter_draft.txt",):
        draft_path = script_dir / filename
        if draft_path.is_file():
            return draft_path.read_text(encoding="utf-8").strip()
    return ""


cover_letter_draft = load_cover_letter_draft()

# ==========================================
# 4. DEFINE TASKS
# ==========================================

class JobMetadata(BaseModel):
    company: str
    position: str
    application_deadline: str | None = None
    entrance_level: str | None = None
    cover_letter_required: str | None = None
    recruiter_name: str | None = None


research_task = Task(
    name="job_research",
    description=(
        "Analyze the following job posting fetched from the URL below:\n\n"
        f"URL: {job_url}\n\n"
        f"Job posting text:\n{job_text}\n\n"
        "Extract the tasks and top 5 mandatory technical skills, "
        "tools, and keywords needed. If the page content seems incomplete or blocked, say so explicitly."
    ),
    expected_output="A clean, bulleted list of tasks and core technical skills, required frameworks, and key terms.",
    agent=job_researcher,
)

metadata_task = Task(
    name="job_metadata",
    description=(
        "Job posting text:\n\n"
        f"{job_text}\n\n"
        "From this page content, extract the job metadata and respond with ONLY a single valid JSON object "
        "(no markdown, no extra text) with exactly these keys:\n"
        '- "company": the hiring company name\n'
        '- "position": the job title\n'
        '- "application_deadline": the application deadline as a string, or null if not stated\n'
        '- "entrance_level": e.g. "entry", "working student", "senior" — or null if unclear\n'
        '- "cover_letter_required": "yes", "no", "optional" or "unknown"\n'
        '- "recruiter_name": the name of the recruiter/contact person if stated, else null\n'
        "Use null for any value you cannot determine."
    ),
    expected_output="A single valid JSON object with the keys company, position, application_deadline, entrance_level, cover_letter_required, recruiter_name.",
    agent=job_researcher,
    context=[research_task],
    output_json=JobMetadata,
)

profile_task = Task(
    name="candidate_profile",
    description=(
        "Read the following text extracted from the candidate's CV and certificates (PDFs). "
        "Produce a structured candidate profile covering: key technical and soft skills, work/internship experience, "
        "education, and all certificates/awards listed. Only use information present in the documents.\n\n"
        f"{candidate_documents}"
    ),
    expected_output=(
        "A structured candidate profile with sections for skills, experience, education, and certifications."
    ),
    agent=profile_analyst
)

fit_task = Task(
    name="candidate_fit",
    description=(
        "Assess how well the candidate fits the job. You are given three inputs as context: "
        "the job metadata (JSON), the job's required skills/keywords, and the candidate profile "
        "extracted from their CV and certificates.\n\n"
        "Compare them honestly and respond in EXACTLY this plain-text format (no JSON, no markdown):\n\n"
        "FIT_RATING: <overall fit as a short string like \"7/10\""
        "MATCHED SKILLS:\n"
        "- <skill/qualification the candidate genuinely meets>\n"
        "- ...\n\n"
        "SKILL GAPS:\n"
        "- <requirement the candidate does NOT meet or has no evidence for>\n"
        "- ...\n\n"
        "FIT SUMMARY:\n"
        "<2-3 sentence honest justification of the rating>"
    ),
    expected_output=(
        "A plain-text fit report with a FIT_RATING line, MATCHED SKILLS list, "
        "SKILL GAPS list, and FIT SUMMARY paragraph."
    ),
    agent=fit_analyst,
    context=[metadata_task, research_task, profile_task]
)

writing_task = Task(
    name="resume_positioning",
    description=(
        "Using the job skills/keywords identified by the research analyst, the candidate profile "
        "produced by the profile analyst (both provided as context) and the fit analysis, write a tailored professional "
        "summary statement and 3 optimized resume bullet points that bridge the candidate's real "
        "background to the job requirements."
    ),
    expected_output="A customized professional summary paragraph followed by 3 high-impact resume bullet points.",
    agent=application_specialist,
    context=[research_task, profile_task, fit_task]
)


# ==========================================
# 5. ASSEMBLE THE CREW
# ==========================================

job_application_crew = Crew(
    agents=[job_researcher, profile_analyst, fit_analyst, application_specialist],
    tasks=[research_task, metadata_task, profile_task, fit_task, writing_task],
    process=Process.sequential,
    verbose=True
)

# ==========================================
# 6. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    print("\n🚀 Initializing Job Application AI Agent Crew with Gemini...\n")
    result = job_application_crew.kickoff()
    
    print("\n==========================================")
    print("✨ FINAL AGENT EXECUTION RESULT ✨")
    print("==========================================\n")
    print(result)

    # ==========================================
    # 7. SAVE JOB APPLICATION TO DATABASE
    # ==========================================
    def task_text(task_output) -> str:
        if task_output is None:
            return ""
        return str(getattr(task_output, "output", task_output))

    outputs_by_name = {}
    for task_output in getattr(result, "tasks_output", []):
        name = getattr(task_output, "name", None)
        if name:
            outputs_by_name[name] = task_output

    job_data = None
    row_id = None
    fit_report = None
    metadata_task_output = outputs_by_name.get("job_metadata")

    if metadata_task_output is None:
        print("\n⚠️  Metadata task output not found; application was not saved.")
    else:
        metadata_text = task_text(metadata_task_output)
        json_dict = getattr(metadata_task_output, "json_dict", None)
        try:
            if isinstance(json_dict, dict):
                job_data = dict(json_dict)
            else:
                match = re.search(r"\{.*\}", metadata_text, re.DOTALL)
                if match:
                    job_data = json.loads(match.group(0))
                else:
                    print("\n⚠️  No JSON object found in the metadata task output.")
        except json.JSONDecodeError as e:
            print(f"\n⚠️  Could not parse job metadata JSON: {e}")
            print(metadata_text[:500])

        if job_data:
            job_data["url"] = job_url
            fit_text = task_text(outputs_by_name.get("candidate_fit"))
            rating_match = re.search(r"FIT_RATING:\s*(.+)", fit_text)
            if rating_match:
                job_data["fit_rating"] = rating_match.group(1).strip()
                fit_report = fit_text.strip()

            try:
                conn = tracker.connect()
                row_id, inserted = tracker.upsert_job(conn, job_data)
                action = "Added new application" if inserted else "Updated existing application"
                print(f"\n💾 {action} (id={row_id}): {job_data.get('company')} — {job_data.get('position')}")
                print(f"Database: {tracker.DB_PATH}")
                if fit_report:
                    print("\n📊 Fit analysis:\n")
                    print(fit_report)
            except (sqlite3.Error, ValueError) as e:
                print(f"\n⚠️  Could not save application to the database: {e}")

    # ==========================================
    # 8. COVER LETTER (only if required by the listing)
    # ==========================================
    cover_letter_required = str(job_data.get("cover_letter_required") or "").lower() if job_data else ""
    if job_data and cover_letter_required in {"yes", "optional", "unknown"}:
        research_out = task_text(outputs_by_name.get("job_research"))
        profile_out = task_text(outputs_by_name.get("candidate_profile"))
        bullets_out = task_text(outputs_by_name.get("resume_positioning"))

        recruiter = job_data.get("recruiter_name")
        salutation = (
            f"Sehr geehrte/r {recruiter},"
            if recruiter
            else "Sehr geehrte Damen und Herren,"
        )

        draft_section = ""
        if cover_letter_draft:
            draft_section = (
                "\n\nCover letter draft provided by the candidate (use it ONLY as inspiration for "
                "tone, phrasing, and structure — adapt the content to THIS job and the candidate "
                "profile; do not copy it verbatim and do not keep claims the profile doesn't support):\n\n"
                f"{cover_letter_draft}"
            )

        cover_letter_task = Task(
description=(
                "Write a cover letter (Anschreiben) in German for the job application below.\n\n"
                "=== TARGET JOB ===\n"
                f"Job details: {json.dumps(job_data, ensure_ascii=False)}\n\n"
                f"Job requirements/keywords:\n{research_out}\n\n"
                "=== CANDIDATE BACKGROUND ===\n"
                f"Candidate profile (from CV and certificates):\n{profile_out}\n\n"
                f"Tailored positioning and resume bullet points:\n{bullets_out}"
                f"{draft_section}\n\n"
                "=== RULES ===\n"
                "- Language: German only.\n"
                f"- Maximum length: 2340 characters (including salutation and sign-off).\n"
                f"- Start with the salutation: \"{salutation}\"\n"
                "- Structure the body in exactly three clearly labeled blocks:\n"
                "  [Sie] — address the company/role and your motivation for applying\n"
                "  [Ich] — present your relevant qualifications and experience\n"
                "  [Wir] — conclude with the mutual fit and a call to action\n"
                "- Write each block as plain prose paragraphs only: no bullet points, numbered lists, "
                "markdown formatting, or headings inside [Sie], [Ich], or [Wir].\n"
                "- In the [Ich] block, address the candidate's competences one by one: for each competence, "
                "explain concretely why it is useful for the tasks of THIS specific job listing, and back it "
                "up with evidence from the candidate profile (e.g. a project, internship, certificate, or "
                "measurable result). Do not list competences without tying them to the job. Argue how the candidate\'s"
                "skills bridge the gaps (if present) to easily master the requirements.\n"
                "- Use placeholders like [Ihr Name] and [Ihre Adresse] where personal details are missing.\n"
                "- Professional, authentic tone; only claim what is supported by the candidate profile."
            ),
            expected_output=(
                "A complete German cover letter under 2340 characters, structured in the three blocks "
                "[Sie], [Ich], and [Wir], written as plain prose without bullet points, including "
                "salutation and sign-off."
            ),
            agent=cover_letter_writer
        )

        cover_letter_crew = Crew(
            agents=[cover_letter_writer],
            tasks=[cover_letter_task],
            process=Process.sequential,
            verbose=True
        )

        print("\n✉️  Cover letter required — generating German Anschreiben...\n")
        letter_result = cover_letter_crew.kickoff()
        letter_text = str(letter_result)

        print("\n==========================================")
        print("✉️  COVER LETTER (German)")
        print("==========================================\n")
        print(letter_text)
        print(f"\n[Length: {len(letter_text)} characters]")