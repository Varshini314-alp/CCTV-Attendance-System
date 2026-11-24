
import sqlite3
import os
import csv
import xml.etree.ElementTree as ET

from datetime import datetime, date, time
from collections import defaultdict


# =========================================================
# 1. PROJECT SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(
    BASE_DIR, "database", "attendance.db"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR, "reports"
)

OFFICE_OPEN = "09:00"
OFFICE_CLOSE = "18:00"

os.makedirs(REPORT_FOLDER, exist_ok=True)


# =========================================================
# 2. DATABASE CONNECTION
# =========================================================

def get_connection():

    conn = sqlite3.connect(
        DATABASE,
        timeout=30
    )

    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# =========================================================
# 3. TIME UTILITIES
# =========================================================

def parse_time(value):

    if not value:
        return None

    for fmt in ("%H:%M:%S", "%H:%M"):

        try:
            return datetime.strptime(
                value, fmt
            )

        except ValueError:
            continue

    return None


def calculate_minutes(start_time, end_time):

    start = parse_time(start_time)
    end = parse_time(end_time)

    if start is None or end is None:
        return 0

    if end < start:
        return 0

    return max(
        0,
        (end - start).total_seconds() / 60
    )


# =========================================================
# 4. GET ATTENDANCE RECORDS
# =========================================================

def get_attendance(report_date=None):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        if report_date:

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
                WHERE date = ?
                ORDER BY employee_id, check_in
            """, (report_date,))

        else:

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
                ORDER BY date DESC, employee_id, check_in
            """)

        return cursor.fetchall()

    finally:
        conn.close()


# =========================================================
# 5. GET EMPLOYEE DETAILS
# =========================================================

def get_employees():

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                employee_id,
                name,
                department,
                phone,
                designation,
                joining_date,
                employment_status
            FROM employees
            ORDER BY employee_id
        """)

        return cursor.fetchall()

    finally:
        conn.close()


# =========================================================
# 6. GET BREAK RECORDS
# =========================================================

def get_breaks(report_date=None):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        # Matches the actual break_records table in db.py.
        # There is no employee_name column in that table.

        if report_date:

            cursor.execute("""
                SELECT
                    employee_id,
                    date,
                    break_start,
                    break_end,
                    duration_minutes,
                    status
                FROM break_records
                WHERE date = ?
                ORDER BY employee_id, break_start
            """, (report_date,))

        else:

            cursor.execute("""
                SELECT
                    employee_id,
                    date,
                    break_start,
                    break_end,
                    duration_minutes,
                    status
                FROM break_records
                ORDER BY date DESC, employee_id, break_start
            """)

        return cursor.fetchall()

    finally:
        conn.close()


# =========================================================
# 7. BREAK CALCULATION
# =========================================================

def calculate_break_minutes(
    break_records,
    report_date=None
):

    total_minutes = 0
    ongoing_count = 0

    today = date.today().isoformat()

    for record in break_records:

        # Actual database column positions:
        # 0 employee_id
        # 1 date
        # 2 break_start
        # 3 break_end
        # 4 duration_minutes
        # 5 status

        break_start = record[2]
        break_end = record[3]
        duration = record[4]
        status = record[5]

        if status == "Ongoing" or not break_end:

            ongoing_count += 1

            # Count elapsed time for an ongoing break
            # only when preparing today's report.

            if report_date == today and break_start:

                elapsed = calculate_minutes(
                    break_start,
                    datetime.now().strftime("%H:%M:%S")
                )

                total_minutes += elapsed

            continue

        if duration is not None:

            total_minutes += max(
                0, float(duration)
            )

        else:

            total_minutes += calculate_minutes(
                break_start,
                break_end
            )

    return round(total_minutes, 2), ongoing_count


# =========================================================
# 8. WORKING HOURS CALCULATION
# =========================================================

def calculate_working_hours(
    attendance_records,
    break_minutes,
    report_date=None
):

    total_minutes = 0

    today = date.today().isoformat()

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    for record in attendance_records:

        # Correct attendance column positions:
        # 0 employee_id
        # 1 employee_name
        # 2 date
        # 3 check_in
        # 4 check_out
        # 5 total_hours
        # 6 status

        check_in = record[3]
        check_out = record[4]

        if not check_in:
            continue

        # For today's open session, use the current time.
        # For older open sessions, do not invent a checkout time.

        if check_out:
            end_time = check_out

        elif report_date == today:
            end_time = current_time

        else:
            continue

        total_minutes += calculate_minutes(
            check_in,
            end_time
        )

    working_minutes = max(
        0,
        total_minutes - break_minutes
    )

    return round(
        working_minutes / 60, 2
    )


# =========================================================
# 9. ATTENDANCE STATUS
# =========================================================

def get_attendance_status(first_check_in):

    if not first_check_in:
        return "Absent"

    arrival = parse_time(first_check_in)

    if arrival is None:
        return "Present"

    office_open = datetime.strptime(
        OFFICE_OPEN, "%H:%M"
    )

    if arrival.time() > office_open.time():
        return "Late"

    return "Present"


# =========================================================
# 10. PREPARE DAILY REPORT DATA
# =========================================================

def prepare_daily_report(report_date=None):

    if report_date is None:
        report_date = date.today().isoformat()

    attendance = get_attendance(report_date)
    breaks = get_breaks(report_date)
    employees = get_employees()

    attendance_by_employee = defaultdict(list)
    breaks_by_employee = defaultdict(list)

    for record in attendance:

        attendance_by_employee[
            str(record[0])
        ].append(record)

    for record in breaks:

        breaks_by_employee[
            str(record[0])
        ].append(record)

    report = []

    for employee in employees:

        employee_id = str(employee[0])
        name = employee[1]
        department = employee[2] or ""
        phone = employee[3] or ""
        designation = employee[4] or ""
        joining_date = employee[5] or ""
        employment_status = employee[6] or "Active"

        if employment_status.lower() != "active":
            continue

        employee_attendance = (
            attendance_by_employee.get(
                employee_id, []
            )
        )

        employee_breaks = (
            breaks_by_employee.get(
                employee_id, []
            )
        )

        # Sort using the actual check-in column.
        employee_attendance.sort(
            key=lambda item: item[3] or ""
        )

        first_check_in = (
            employee_attendance[0][3]
            if employee_attendance else ""
        )

        last_check_out = (
            employee_attendance[-1][4]
            if employee_attendance else ""
        )

        if employee_attendance:

            last_session = employee_attendance[-1]

            current_status = (
                "Currently Inside"
                if last_session[4] is None
                else "Logged Out"
            )

        else:

            current_status = "Absent"

        break_minutes, ongoing_breaks = (
            calculate_break_minutes(
                employee_breaks,
                report_date
            )
        )

        working_hours = calculate_working_hours(
            employee_attendance,
            break_minutes,
            report_date
        )

        attendance_status = get_attendance_status(
            first_check_in
        )

        if not employee_attendance:
            attendance_status = "Absent"

        report.append({

            "employee_id": employee_id,
            "name": name,
            "department": department,
            "phone": phone,
            "designation": designation,
            "joining_date": joining_date,
            "date": report_date,

            "check_in": first_check_in,
            "check_out": last_check_out,

            "session_count": len(
                employee_attendance
            ),

            "break_minutes": break_minutes,
            "ongoing_breaks": ongoing_breaks,

            "working_hours": working_hours,

            "attendance_status": attendance_status,
            "current_status": current_status,

            "sessions": employee_attendance

        })

    return report


# =========================================================
# 11. GENERATE DAILY XML
# =========================================================

def generate_daily_xml(report_date=None):

    if report_date is None:
        report_date = date.today().isoformat()

    report = prepare_daily_report(report_date)

    root = ET.Element(
        "DailyAttendance",
        date=report_date,
        generated_at=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    total_employees = len(report)

    present_count = sum(
        1 for item in report
        if item["attendance_status"] != "Absent"
    )

    absent_count = (
        total_employees - present_count
    )

    late_count = sum(
        1 for item in report
        if item["attendance_status"] == "Late"
    )

    inside_count = sum(
        1 for item in report
        if item["current_status"] == "Currently Inside"
    )

    ET.SubElement(
        root, "TotalEmployees"
    ).text = str(total_employees)

    ET.SubElement(
        root, "PresentCount"
    ).text = str(present_count)

    ET.SubElement(
        root, "AbsentCount"
    ).text = str(absent_count)

    ET.SubElement(
        root, "LateCount"
    ).text = str(late_count)

    ET.SubElement(
        root, "CurrentlyInside"
    ).text = str(inside_count)

    attendance_records = ET.SubElement(
        root, "AttendanceRecords"
    )

    for item in report:

        employee = ET.SubElement(
            attendance_records, "Employee"
        )

        fields = {

            "EmployeeID": item["employee_id"],
            "EmployeeName": item["name"],
            "Department": item["department"],
            "Designation": item["designation"],
            "JoiningDate": item["joining_date"],
            "Date": item["date"],
            "CheckIn": item["check_in"],
            "CheckOut": item["check_out"],
            "SessionCount": item["session_count"],
            "BreakDurationMinutes": item["break_minutes"],
            "OngoingBreakCount": item["ongoing_breaks"],
            "TotalWorkingHours": item["working_hours"],
            "AttendanceStatus": item["attendance_status"],
            "CurrentStatus": item["current_status"]

        }

        for tag, value in fields.items():

            ET.SubElement(
                employee, tag
            ).text = str(value)

        sessions = ET.SubElement(
            employee, "Sessions"
        )

        for session_number, session in enumerate(
            item["sessions"], start=1
        ):

            session_element = ET.SubElement(
                sessions, "Session"
            )

            ET.SubElement(
                session_element, "SessionNumber"
            ).text = str(session_number)

            ET.SubElement(
                session_element, "LoginTime"
            ).text = session[3] or ""

            ET.SubElement(
                session_element, "LogoutTime"
            ).text = session[4] or ""

    tree = ET.ElementTree(root)

    ET.indent(
        tree,
        space="    ",
        level=0
    )

    filename = (
        f"attendance_{report_date}.xml"
    )

    file_path = os.path.join(
        REPORT_FOLDER, filename
    )

    tree.write(
        file_path,
        encoding="utf-8",
        xml_declaration=True
    )

    print(
        "[XML] Daily attendance report updated:",
        file_path
    )

    return file_path


# =========================================================
# 12. DISPLAY ATTENDANCE REPORT
# =========================================================

def display_report(report_date=None):

    report = prepare_daily_report(report_date)

    print("\n======================================")
    print("       EMPLOYEE ATTENDANCE REPORT")
    print("======================================")

    if not report:

        print("No employee records found.")
        return

    for item in report:

        print("--------------------------------------")

        print(
            "Employee ID:",
            item["employee_id"]
        )

        print(
            "Employee Name:",
            item["name"]
        )

        print(
            "Department:",
            item["department"]
        )

        print(
            "Designation:",
            item["designation"]
        )

        print(
            "Date:",
            item["date"]
        )

        print(
            "Check-in:",
            item["check_in"] or "Absent"
        )

        print(
            "Check-out:",
            item["check_out"] or "Pending"
        )

        print(
            "Working Hours:",
            item["working_hours"]
        )

        print(
            "Break Minutes:",
            item["break_minutes"]
        )

        print(
            "Ongoing Breaks:",
            item["ongoing_breaks"]
        )

        print(
            "Attendance:",
            item["attendance_status"]
        )

        print(
            "Current Status:",
            item["current_status"]
        )

    print("--------------------------------------")

    print(
        "Total employees:",
        len(report)
    )


# =========================================================
# 13. EXPORT CSV REPORT
# =========================================================

def export_report(report_date=None):

    report = prepare_daily_report(report_date)

    if not report:

        print("No records available to export!")
        return None

    filename = (
        "attendance_report_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".csv"
    )

    file_path = os.path.join(
        REPORT_FOLDER, filename
    )

    headers = [

        "Employee ID",
        "Employee Name",
        "Department",
        "Designation",
        "Joining Date",
        "Date",
        "Check-in",
        "Check-out",
        "Session Count",
        "Break Minutes",
        "Ongoing Breaks",
        "Working Hours",
        "Attendance Status",
        "Current Status"

    ]

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        for item in report:

            writer.writerow([

                item["employee_id"],
                item["name"],
                item["department"],
                item["designation"],
                item["joining_date"],
                item["date"],
                item["check_in"],
                item["check_out"],
                item["session_count"],
                item["break_minutes"],
                item["ongoing_breaks"],
                item["working_hours"],
                item["attendance_status"],
                item["current_status"]

            ])

    print(
        "[CSV] Report exported:",
        file_path
    )

    return file_path


# =========================================================
# 14. MAIN MENU
# =========================================================

if __name__ == "__main__":

    while True:

        print("\n================================")
        print("       ATTENDANCE REPORT")
        print("================================")

        print("1. View Today's Attendance")
        print("2. Export Today's Attendance CSV")
        print("3. Generate Today's XML")
        print("4. Exit")

        choice = input(
            "Enter your choice (1/2/3/4): "
        ).strip()

        if choice == "1":

            display_report()

        elif choice == "2":

            export_report()

        elif choice == "3":

            path = generate_daily_xml()

            print("XML generated:", path)

        elif choice == "4":

            print(
                "Exiting Attendance Report..."
            )

            break

        else:

            print("Invalid choice!")
