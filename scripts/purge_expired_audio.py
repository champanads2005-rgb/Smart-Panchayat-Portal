"""
Purges original voice-complaint recordings older than
config.Config.AUDIO_RETENTION_DAYS. Meant to be run periodically (e.g.
a daily cron job) - see the crontab example in ml/asr/data/README.md.

What this does and does NOT delete:
    - DELETES: the original audio file on disk, and its
      complaint_audio row.
    - DOES NOT touch: the complaint itself, or complaints.description -
      the confirmed transcript was copied there at submission time and
      is permanent complaint history independent of the original
      recording's retention policy.

Run:
    python scripts/purge_expired_audio.py            # deletes as configured
    python scripts/purge_expired_audio.py --dry-run   # only reports what would be deleted
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from app import get_db_connection  # reuses the exact same DB connection helper as app.py


def purge_expired_audio(dry_run: bool = False) -> dict:
    if Config.AUDIO_RETENTION_DAYS <= 0:
        print("AUDIO_RETENTION_DAYS is 0 or unset - auto-purging is disabled. Nothing to do.")
        return {"deleted_rows": 0, "deleted_files": 0, "missing_files": 0}

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT id, stored_filename
        FROM complaint_audio
        WHERE uploaded_at < (NOW() - INTERVAL %s DAY)
        """,
        (Config.AUDIO_RETENTION_DAYS,),
    )
    expired_rows = cursor.fetchall()

    deleted_files = 0
    missing_files = 0

    for row in expired_rows:
        file_path = os.path.join(Config.AUDIO_UPLOAD_FOLDER, row["stored_filename"])
        if os.path.isfile(file_path):
            if not dry_run:
                os.remove(file_path)
            deleted_files += 1
        else:
            missing_files += 1

        if not dry_run:
            cursor.execute("DELETE FROM complaint_audio WHERE id = %s", (row["id"],))

    if not dry_run:
        db.commit()
    cursor.close()
    db.close()

    verb = "Would delete" if dry_run else "Deleted"
    print(
        f"{verb} {len(expired_rows)} complaint_audio row(s) older than "
        f"{Config.AUDIO_RETENTION_DAYS} day(s) "
        f"({deleted_files} file(s) removed, {missing_files} already missing on disk)."
    )
    return {
        "deleted_rows": len(expired_rows),
        "deleted_files": deleted_files,
        "missing_files": missing_files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting anything.")
    args = parser.parse_args()
    purge_expired_audio(dry_run=args.dry_run)
