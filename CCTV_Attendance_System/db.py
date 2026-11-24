
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "database", "attendance.db")


def get_connection():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    return sqlite3.connect(DATABASE)


def create_attendance_table():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            date TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            total_hours REAL,
            status TEXT DEFAULT 'Present'
        )
    """)

    # Add total_hours to an existing table if it is missing.
    cursor.execute("PRAGMA table_info(attendance)")
    columns = [column[1] for column in cursor.fetchall()]

    if "total_hours" not in columns:
        cursor.execute("""
            ALTER TABLE attendance
            ADD COLUMN total_hours REAL
        """)

    conn.commit()
    conn.close()

    print("Attendance table created successfully!")


if __name__ == "__main__":
    create_attendance_table()
