# Multi-Agent Job Application Pipeline

## Overview

A multi-agent crew compares a job listing with a candidate's profile, identifies matches, gaps and rates the fit.
An application tracker stores the job listing in a database file. Furthermore, a cover letter is written, if required or optional.
All in a single automated pipeline.


## Features

* **Profiling** of the Job and the candidate. A research agent extracts tasks and requirements from the job listing. Another profile agent creates a candidate's profile, based on documents provided by the candidate (e.g. CV and certificates).
* **Objective Fit Assessment** between the job listing and the candidate. A fit agent compares the job listing with the candidate's profile, identifies matches, gaps and rates the match with a number between zero and ten, being honest and objective.
* **Strategic Positioning** to bridge the job listing and the candidate's profile provided by an application specialist.
* **Automatic Tracking** of the applications. The metadata of the applications is stored in a database to track applications. Status and date of application can be updated afterwards.
* **Cover Letter Generation** if required or optional based on the job listing. You can provide your own cover letter in the project folder, to give it a personal touch.


## Tech Stack

* **Python-Based Multi Ai Agent Collaboration** via **CrewAI**.
* **Create Database** to store applications via **SQLite3** and **JSON**.
* **Web Fetching** to transform HTML into plain readable text via **HTMLParser**.
* **Read PDF** files to extract text content via **PDFplumber**.


## How to run

1. Ensure all dependencies are installed.
2. Set up your API keys in a `.env` file in the project folder.
3. Drop your CV and other relevant documents in the project folder.
4. For a personal touch of the cover letter, drop 'cover_letter_draft.txt' in the project folder. Default language of cover letter is German. Adjust the language in the code if needed.
5. Run `python application_pipeline.py` to start the application pipeline.
6. Review the generated reports and database entries.