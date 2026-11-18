"""SQLite-backed job application tracker.

Stores one row per job application in `applications.db` (same folder as the project).
- Upserts by URL: an existing URL is overwritten, a new URL inserts a new row.
- Extra columns can be added at runtime via `ensure_column()` / the CLI `add-column`.
- Use the CLI (`python tracker.py ...`) or import the functions to refine entries.
"""

import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "applications.db"

CORE_COLUMNS = {
    "company": "TEXT",
    "position": "TEXT",
    "url": "TEXT",
    "application_deadline": "TEXT",
    "entrance_level": "TEXT",
    "cover_letter_required": "TEXT",
    "recruiter_name": "TEXT",
    "fit_rating": "TEXT",
    "application_status": "TEXT DEFAULT 'not applied'",
    "application_date": "TEXT",
}



_COLUMN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COLUMN_TYPE_RE = re.compile(
    r"^(?:TEXT|INTEGER|REAL|BLOB|NUMERIC|BOOLEAN|DATE|DATETIME|TIMESTAMP|"
    r"VARCHAR|CHAR|NVARCHAR|NCHAR|DECIMAL|FLOAT|DOUBLE|INT|BIGINT|SMALLINT)"
    r"(?:\s*\([0-9, ]+\))?$",
    re.IGNORECASE,
)


def _is_valid_column_name(name: object) -> bool:
    return isinstance(name, str) and bool(_COLUMN_NAME_RE.fullmatch(name))


def _is_valid_column_type(sql_type: object) -> bool:
    return isinstance(sql_type, str) and bool(
        _COLUMN_TYPE_RE.fullmatch(sql_type.strip())
    )


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    columns = "id INTEGER PRIMARY KEY AUTOINCREMENT, " + ", ".join(
        f"{name} {sql_type}" for name, sql_type in CORE_COLUMNS.items()
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS applications ({columns})")
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(applications)")}
    schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
    # Ensure all core columns exist even in databases created before they were added
    for name, sql_type in CORE_COLUMNS.items():
        if name not in existing:
            conn.execute(
                f"ALTER TABLE applications ADD COLUMN {_quote_identifier(name)} {sql_type}"
            )
    # The job's identity is its URL: one row per unique URL 
    conn.execute("DROP INDEX IF EXISTS idx_app_company_position")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_app_url "
        "ON applications (lower(url))"
    )
    if schema_version < 1:
        conn.execute("PRAGMA user_version = 1")
    conn.commit()


def ensure_column(conn: sqlite3.Connection, name: str, sql_type: str = "TEXT") -> None:
    """Adds a new column to the applications table if it doesn't exist yet."""
    if not _is_valid_column_name(name):
        raise ValueError(f"Invalid column name: {name!r}")
    if not _is_valid_column_type(sql_type):
        raise ValueError(f"Invalid column type: {sql_type!r}")

    init_db(conn)
    existing = {row["name"].lower() for row in conn.execute("PRAGMA table_info(applications)")}
    if name.lower() not in existing:
        conn.execute(
            f"ALTER TABLE applications ADD COLUMN {_quote_identifier(name)} {sql_type.strip()}"
        )
        conn.commit()


def _normalize(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def upsert_job(conn: sqlite3.Connection, job: dict) -> tuple[int, bool]:
    """Insert a job, or update the existing row.

    The job's identity is its URL: a listing whose URL already exists in the
    database (case-insensitive) is overwritten; any other URL — or a missing
    URL with an unseen (company, position) pair — creates a new row.

    Returns (row_id, was_inserted).
    """
    init_db(conn)
    company = _normalize(job.get("company"))
    position = _normalize(job.get("position"))
    if not company or not position:
        raise ValueError("Job must include at least 'company' and 'position'.")

    url = _normalize(job.get("url"))

    if url:
        row = conn.execute(
            "SELECT id FROM applications WHERE lower(url) = lower(?)",
            (url,),
        ).fetchone()
    else:
        # No URL to key on — fall back to (company, position)
        row = conn.execute(
            "SELECT id FROM applications WHERE lower(company) = lower(?) "
            "AND lower(position) = lower(?)",
            (company, position),
        ).fetchone()



    fields = {
        "company": company,
        "position": position,
        "url": url,
        "application_deadline": _normalize(job.get("application_deadline")),
        "entrance_level": _normalize(job.get("entrance_level")),
        "cover_letter_required": _normalize(job.get("cover_letter_required")),
        "recruiter_name": _normalize(job.get("recruiter_name")),
        "fit_rating": _normalize(job.get("fit_rating")),
        "application_status": _normalize(job.get("application_status")),
        "application_date": _normalize(job.get("application_date")),
    }
    # Optional extra fields become columns automatically
    reserved_names = {name.lower() for name in CORE_COLUMNS} | {"id"}
    used_names = {name.lower() for name in fields}
    for key, value in job.items():
        if not isinstance(key, str) or value is None:
            continue
        key_lower = key.lower()
        if (
            key_lower in reserved_names
            or key_lower in used_names
            or not _is_valid_column_name(key)
        ):
            continue
        ensure_column(conn, key)
        fields[key] = _normalize(value)
        used_names.add(key_lower)

    if row:
        # Only overwrite fields that were actually provided (non-None)
        update_fields = {k: v for k, v in fields.items() if v is not None}
        if update_fields:
            assignments = ", ".join(f"{_quote_identifier(k)} = ?" for k in update_fields)
            conn.execute(
                f"UPDATE applications SET {assignments} WHERE id = ?",
                (*update_fields.values(), row["id"]),
            )
            conn.commit()
        return row["id"], False

    if fields["application_status"] is None:
        fields["application_status"] = "not applied"
    columns = ", ".join(_quote_identifier(name) for name in fields)
    placeholders = ", ".join("?" for _ in fields)
    cursor = conn.execute(
        f"INSERT INTO applications ({columns}) VALUES ({placeholders})", tuple(fields.values())
    )
    conn.commit()
    return cursor.lastrowid, True


def update_status(conn: sqlite3.Connection, company: str, position: str, status: str) -> bool:
    init_db(conn)
    company = _normalize(company)
    position = _normalize(position)
    status = _normalize(status)
    if not company or not position or not status:
        return False
    cur = conn.execute(
        """UPDATE applications SET application_status = ?
           WHERE lower(company) = lower(?) AND lower(position) = lower(?)""",
        (status, company, position),
    )
    conn.commit()
    return cur.rowcount > 0


def list_jobs(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    init_db(conn)
    if status:
        rows = conn.execute(
            "SELECT * FROM applications WHERE application_status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM applications ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No applications found.")
        return
    columns = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c) or "")) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for r in rows:
        print(" | ".join(str(r.get(c) or "").ljust(widths[c]) for c in columns))


if __name__ == "__main__":
    conn = connect()
    init_db(conn)

    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:")
        print("  python tracker.py list [status]")
        print("  python tracker.py update <company> <position> <new_status>")
        print("  python tracker.py add-column <name> [SQL_TYPE]")
        sys.exit(0)

    command = sys.argv[1]

    if command == "list":
        status = sys.argv[2] if len(sys.argv) > 2 else None
        _print_table(list_jobs(conn, status))
    elif command == "update":
        if len(sys.argv) < 5:
            sys.exit("Usage: python tracker.py update <company> <position> <new_status>")
        ok = update_status(conn, sys.argv[2], " ".join(sys.argv[3:-1]), sys.argv[-1])
        print("Updated." if ok else "No matching application found.")
    elif command == "add-column":
        if len(sys.argv) < 3:
            sys.exit("Usage: python tracker.py add-column <name> [SQL_TYPE]")
        ensure_column(conn, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "TEXT")
        print(f"Column '{sys.argv[2]}' ensured.")
    else:
        sys.exit(f"Unknown command: {command}")
