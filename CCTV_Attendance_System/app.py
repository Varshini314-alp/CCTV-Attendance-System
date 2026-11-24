
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, make_response
)

import os
import sqlite3
import csv
import io
import json
import base64
import uuid
import xml.etree.ElementTree as ET

from datetime import date
from werkzeug.utils import secure_filename


app = Flask(__name__)


# -----------------------------------------
# PROJECT PATHS
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "employees"
)

DATABASE = os.path.join(
    BASE_DIR, "database", "attendance.db"
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.dirname(DATABASE), exist_ok=True)


# -----------------------------------------
# DATABASE CONNECTION
# -----------------------------------------

def get_db():

    conn = sqlite3.connect(DATABASE)
    return conn


# -----------------------------------------
# CREATE EMPLOYEE TABLE
# -----------------------------------------

def create_table():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            phone TEXT NOT NULL,
            photo TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# -----------------------------------------
# SERVE EMPLOYEE PHOTOS
# -----------------------------------------

@app.route("/employee_photo/<path:filename>")
def employee_photo(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# -----------------------------------------
# HOME - EMPLOYEE REGISTRATION
# -----------------------------------------

@app.route("/")
def home():

    return render_template("register.html")


# -----------------------------------------
# REGISTER EMPLOYEE
# -----------------------------------------

@app.route("/register", methods=["POST"])
def register():

    employee_id = request.form.get(
        "employee_id", ""
    ).strip()

    name = request.form.get(
        "name", ""
    ).strip()

    department = request.form.get(
        "department", ""
    ).strip()

    phone = request.form.get(
        "phone", ""
    ).strip()

    photo = request.files.get("photo")

    captured_data = request.form.get(
        "captured_photos", ""
    )

    if not all([employee_id, name, department, phone]):
        return "Please fill all employee details."

    if not photo or not photo.filename:
        return "Please upload an employee photo."

    safe_employee_id = secure_filename(employee_id)

    if not safe_employee_id:
        return "Invalid employee ID."

    original_filename = secure_filename(photo.filename)

    if not original_filename:
        return "Invalid photo filename."

    extension = os.path.splitext(original_filename)[1].lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        return "Please upload a JPG, JPEG or PNG photo."

    try:

        captured_photos = json.loads(captured_data)

        if not isinstance(captured_photos, list):
            return "Invalid training photos."

        if not 3 <= len(captured_photos) <= 10:
            return "Please capture between 3 and 10 training photos."

        training_images = []

        for image_data in captured_photos:

            if not isinstance(image_data, str):
                return "Invalid camera photo."

            if not image_data.startswith("data:image/"):
                return "Invalid camera image format."

            header, encoded_image = image_data.split(",", 1)

            if ";base64" not in header:
                return "Invalid camera image encoding."

            image_bytes = base64.b64decode(
                encoded_image,
                validate=True
            )

            if not image_bytes:
                return "Empty training photo."

            if len(image_bytes) > 5 * 1024 * 1024:
                return "Each training photo must be under 5 MB."

            training_images.append(image_bytes)

    except Exception:

        return "Could not process camera photos. Please capture them again."

    unique_id = uuid.uuid4().hex

    filename = (
        safe_employee_id + "_" +
        unique_id + extension
    )

    photo_path = os.path.join(
        "employees",
        filename
    )

    full_photo_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    training_folder = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "training_photos",
        safe_employee_id
    )

    saved_files = []

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO employees
            (employee_id, name, department, phone, photo)
            VALUES (?, ?, ?, ?, ?)
        """, (
            employee_id,
            name,
            department,
            phone,
            photo_path
        ))

        photo.save(full_photo_path)
        saved_files.append(full_photo_path)

        os.makedirs(training_folder, exist_ok=True)

        for index, image_bytes in enumerate(
            training_images, start=1
        ):

            training_filename = (
                safe_employee_id + "_" +
                str(index) + "_" +
                unique_id + ".jpg"
            )

            training_path = os.path.join(
                training_folder,
                training_filename
            )

            with open(training_path, "wb") as image_file:
                image_file.write(image_bytes)

            saved_files.append(training_path)

        conn.commit()

    except sqlite3.IntegrityError:

        conn.rollback()
        conn.close()

        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        return "Employee ID already exists!"

    except Exception as error:

        conn.rollback()

        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        conn.close()

        print("Registration error:", error)

        return "Employee registration failed."

    conn.close()

    return redirect(url_for("employee_list"))


# -----------------------------------------
# DISPLAY ALL EMPLOYEES
# -----------------------------------------

@app.route("/employees")
def employee_list():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM employees
        ORDER BY name
    """)

    employees = cursor.fetchall()

    conn.close()

    return render_template(
        "employees.html",
        employees=employees
    )


# -----------------------------------------
# ATTENDANCE DASHBOARD
# -----------------------------------------

@app.route("/attendance")
def attendance_dashboard():

    conn = get_db()
    cursor = conn.cursor()

    today = date.today().isoformat()

    search_query = request.args.get(
        "search", ""
    ).strip()

    search_pattern = f"%{search_query}%"

    cursor.execute("""
        SELECT COUNT(*)
        FROM employees
    """)

    total_employees = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT employee_id)
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
    """, (today,))

    present = cursor.fetchone()[0]

    absent = max(
        0,
        total_employees - present
    )

    cursor.execute("""
        SELECT COUNT(DISTINCT employee_id)
        FROM attendance
        WHERE date = ?
        AND check_in IS NOT NULL
        AND check_out IS NULL
    """, (today,))

    currently_inside = cursor.fetchone()[0]

    cursor.execute("""
        SELECT
            e.employee_id,
            e.name,
            COALESCE(a.date, ?) AS attendance_date,
            a.check_in,
            a.check_out,
            a.total_hours,
            COALESCE(a.status, 'Absent') AS status,
            e.photo

        FROM employees e

        LEFT JOIN attendance a
            ON e.employee_id = a.employee_id
            AND a.date = ?

        WHERE
            e.employee_id LIKE ?
            OR e.name LIKE ?

        ORDER BY e.name
    """, (
        today,
        today,
        search_pattern,
        search_pattern
    ))

    attendance_records = cursor.fetchall()

    conn.close()

    return render_template(
        "attendance.html",
        total_employees=total_employees,
        present=present,
        absent=absent,
        currently_inside=currently_inside,
        attendance_records=attendance_records,
        today=today,
        search_query=search_query
    )


# -----------------------------------------
# GET DAILY OR MONTHLY REPORT DATA
# -----------------------------------------

def get_report_data(report_type, selected_date, selected_month):

    conn = get_db()
    cursor = conn.cursor()

    if report_type == "monthly":

        cursor.execute("""
            SELECT
                e.employee_id,
                e.name,
                COUNT(DISTINCT a.date) AS present_days,
                ROUND(
                    COALESCE(SUM(a.total_hours), 0), 2
                ) AS total_hours,

                CASE
                    WHEN COUNT(a.id) > 0 THEN 'Present'
                    ELSE 'Absent'
                END AS status

            FROM employees e

            LEFT JOIN attendance a
                ON e.employee_id = a.employee_id
                AND strftime('%Y-%m', a.date) = ?
                AND a.status = 'Present'

            GROUP BY
                e.employee_id,
                e.name

            ORDER BY e.name

        """, (selected_month,))

        headers = [
            "Employee ID",
            "Employee Name",
            "Present Days",
            "Total Working Hours",
            "Status"
        ]

    else:

        cursor.execute("""
            SELECT
                e.employee_id,
                e.name,
                ? AS date,
                a.check_in,
                a.check_out,
                a.total_hours,
                COALESCE(a.status, 'Absent') AS status

            FROM employees e

            LEFT JOIN attendance a
                ON e.employee_id = a.employee_id
                AND a.date = ?

            ORDER BY e.name

        """, (
            selected_date,
            selected_date
        ))

        headers = [
            "Employee ID",
            "Employee Name",
            "Date",
            "Check-in",
            "Check-out",
            "Total Working Hours",
            "Status"
        ]

    records = cursor.fetchall()

    conn.close()

    return headers, records


# -----------------------------------------
# ATTENDANCE REPORT PAGE
# -----------------------------------------

@app.route("/reports")
def attendance_reports():

    report_type = request.args.get(
        "type", "daily"
    )

    if report_type not in ["daily", "monthly"]:
        report_type = "daily"

    selected_date = request.args.get(
        "date", date.today().isoformat()
    )

    selected_month = request.args.get(
        "month", date.today().strftime("%Y-%m")
    )

    headers, records = get_report_data(
        report_type,
        selected_date,
        selected_month
    )

    return render_template(
        "reports.html",
        report_type=report_type,
        selected_date=selected_date,
        selected_month=selected_month,
        headers=headers,
        records=records
    )


# -----------------------------------------
# EXPORT ATTENDANCE REPORT TO CSV
# -----------------------------------------

@app.route("/reports/export")
def export_attendance_report():

    report_type = request.args.get(
        "type", "daily"
    )

    if report_type not in ["daily", "monthly"]:
        report_type = "daily"

    selected_date = request.args.get(
        "date", date.today().isoformat()
    )

    selected_month = request.args.get(
        "month", date.today().strftime("%Y-%m")
    )

    headers, records = get_report_data(
        report_type,
        selected_date,
        selected_month
    )

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow(headers)

    for record in records:
        writer.writerow(record)

    response = make_response(
        "\ufeff" + output.getvalue()
    )

    response.headers["Content-Disposition"] = (
        f"attachment; filename={report_type}_attendance_report.csv"
    )

    response.headers["Content-Type"] = (
        "text/csv; charset=utf-8"
    )

    return response


# -----------------------------------------
# EXPORT CCTV ATTENDANCE REPORT TO XML
# -----------------------------------------

@app.route("/reports/export/xml")
def export_attendance_xml():

    selected_date = request.args.get(
        "date", date.today().isoformat()
    )

    # Validate date
    try:
        date.fromisoformat(selected_date)
    except ValueError:
        return "Invalid report date.", 400

    conn = get_db()
    cursor = conn.cursor()

    # Get all employees and attendance details
    cursor.execute("""
        SELECT
            e.employee_id,
            e.name,
            e.department,
            a.date,
            a.check_in,
            a.check_out,
            a.total_hours,
            COALESCE(a.status, 'Absent')
        FROM employees e

        LEFT JOIN attendance a
            ON e.employee_id = a.employee_id
            AND a.date = ?

        ORDER BY e.name
    """, (selected_date,))

    employees = cursor.fetchall()

    # Create XML root
    root = ET.Element("CCTVAttendanceReport")

    root.set("date", selected_date)
    root.set("office_opening_time", "09:00")
    root.set("office_closing_time", "18:00")

    for employee in employees:

        employee_id = employee[0]

        employee_element = ET.SubElement(
            root, "Employee"
        )

        fields = [
            ("EmployeeID", employee[0]),
            ("EmployeeName", employee[1]),
            ("Department", employee[2]),
            ("Date", employee[3] or selected_date),
            ("CheckIn", employee[4]),
            ("CheckOut", employee[5]),
            ("RecordedWorkingHours", employee[6]),
            ("AttendanceStatus", employee[7])
        ]

        for tag, value in fields:

            element = ET.SubElement(
                employee_element, tag
            )

            element.text = (
                str(value) if value is not None else ""
            )

        # Get break records
        cursor.execute("""
            SELECT
                break_start,
                break_end,
                duration_minutes,
                status
            FROM break_records
            WHERE employee_id = ?
            AND date = ?
            ORDER BY break_start
        """, (employee_id, selected_date))

        breaks = cursor.fetchall()

        breaks_element = ET.SubElement(
            employee_element, "Breaks"
        )

        total_break_minutes = 0

        for break_record in breaks:

            break_element = ET.SubElement(
                breaks_element, "Break"
            )

            break_fields = [
                ("BreakOut", break_record[0]),
                ("BreakIn", break_record[1]),
                ("DurationMinutes", break_record[2]),
                ("Status", break_record[3])
            ]

            for tag, value in break_fields:

                element = ET.SubElement(
                    break_element, tag
                )

                element.text = (
                    str(value)
                    if value is not None else ""
                )

            if break_record[2] is not None:

                total_break_minutes += break_record[2]

        total_break_element = ET.SubElement(
            employee_element, "TotalBreakMinutes"
        )

        total_break_element.text = str(
            round(total_break_minutes, 2)
        )

    conn.close()

    # Generate XML content
    xml_data = ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True
    )

    response = make_response(xml_data)

    response.headers["Content-Type"] = (
        "application/xml; charset=utf-8"
    )

    response.headers["Content-Disposition"] = (
        f"attachment; filename=cctv_attendance_{selected_date}.xml"
    )

    return response


# -----------------------------------------
# RUN APPLICATION
# -----------------------------------------

if __name__ == "__main__":

    create_table()

    app.run(
        debug=True
    )
