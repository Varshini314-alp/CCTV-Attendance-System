
import sqlite3
import os
import csv
from datetime import datetime

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(
    BASE_DIR, "database", "attendance.db"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR, "reports"
)

os.makedirs(REPORT_FOLDER, exist_ok=True)


# =====================================
# GET ATTENDANCE RECORDS
# =====================================

def get_attendance():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            employee_id,
            employee_name,
            date,
            check_in,
            check_out,
            total_hours,
            status
        FROM attendance
        ORDER BY date DESC, check_in DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return records


# =====================================
# DISPLAY ATTENDANCE REPORT
# =====================================

def display_report():

    records = get_attendance()

    print("\n======================================")
    print("       EMPLOYEE ATTENDANCE REPORT")
    print("======================================")

    if not records:
        print("No attendance records found!")
        return

    for record in records:

        print("--------------------------------------")
        print("Employee ID:", record[0])
        print("Employee Name:", record[1])
        print("Date:", record[2])
        print("Check-in:", record[3])
        print("Check-out:", record[4] or "Not recorded")
        print(
            "Working Hours:",
            record[5] if record[5] is not None else "Pending"
        )
        print("Status:", record[6])

    print("--------------------------------------")
    print("Total attendance records:", len(records))


# =====================================
# EXPORT ATTENDANCE TO CSV
# =====================================

def export_report():

    records = get_attendance()

    if not records:
        print("No records available to export!")
        return

    filename = (
        "attendance_report_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )

    file_path = os.path.join(
        REPORT_FOLDER, filename
    )

    headers = [
        "Employee ID",
        "Employee Name",
        "Date",
        "Check-in",
        "Check-out",
        "Total Working Hours",
        "Status"
    ]

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerows(records)

    print("\nReport exported successfully!")
    print("File location:", file_path)


# =====================================
# MAIN MENU
# =====================================

if __name__ == "__main__":

    while True:

        print("\n================================")
        print("       ATTENDANCE REPORT")
        print("================================")

        print("1. View Attendance Report")
        print("2. Export Attendance to CSV")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":

            display_report()

        elif choice == "2":

            export_report()

        elif choice == "3":

            print("Exiting Attendance Report...")
            break

        else:

            print("Invalid choice! Try again.")
