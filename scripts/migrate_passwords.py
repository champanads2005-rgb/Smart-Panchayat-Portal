"""
One-time migration: hashes any plaintext passwords in the `users` table.

Safe to run multiple times - rows that already look like a Werkzeug hash
are skipped.

Run from the project root:
    python scripts/migrate_passwords.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mysql.connector

from config import Config
from security_utils import hash_password, looks_like_hash


def main():
    db = mysql.connector.connect(
        host=Config.DB_HOST,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        database=Config.DB_NAME,
    )
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, email, password FROM users")
    users = cursor.fetchall()

    update_cursor = db.cursor()
    migrated = 0
    skipped = 0

    for user in users:
        if looks_like_hash(user["password"]):
            skipped += 1
            continue
        new_hash = hash_password(user["password"])
        update_cursor.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (new_hash, user["id"]),
        )
        migrated += 1
        print(f"Migrated password for user id={user['id']} email={user['email']}")

    db.commit()
    update_cursor.close()
    cursor.close()
    db.close()

    print(f"\nDone. Migrated: {migrated}, already hashed (skipped): {skipped}")


if __name__ == "__main__":
    main()
