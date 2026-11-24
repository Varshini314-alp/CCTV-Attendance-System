
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, make_response
)
import os
import sqlite3
import csv
import io
from datetime import date
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -----------------------------------------
# PROJECT PATHS
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "employees")

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

    if not all([employee_id, name, department, phone]):
        return "Please fill all employee details."

    if not photo or not photo.filename:
        return "Please upload an employee photo."

    filename = secure_filename(
        employee_id + "_" + photo.filename
    )

    if not filename:
        return "Invalid photo filename."

    photo_path = os.path.join(
        "employees",
        filename
    )

    full_photo_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

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

        conn.commit()

    except sqlite3.IntegrityError:

        conn.close()
        return "Employee ID already exists!"

    except Exception:

        conn.rollback()
        conn.close()
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

    # Total employees

    cursor.execute("""
        SELECT COUNT(*)
        FROM employees
    """)

    total_employees = cursor.fetchone()[0]

    # Present employees today

    cursor.execute("""
        SELECT COUNT(DISTINCT employee_id)
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
    """, (today,))

    present = cursor.fetchone()[0]

    # Absent employees

    absent = max(
        0,
        total_employees - present
    )

    # Employees currently inside

    cursor.execute("""
        SELECT COUNT(DISTINCT employee_id)
        FROM attendance
        WHERE date = ?
        AND check_in IS NOT NULL
        AND check_out IS NULL
    """, (today,))

    currently_inside = cursor.fetchone()[0]

    # Attendance list with employee photos

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
# RUN APPLICATION
# -----------------------------------------

if __name__ == "__main__":

    create_table()

    app.run(
        debug=True
    )
