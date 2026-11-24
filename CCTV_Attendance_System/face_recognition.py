
import cv2
import os
import json
import time
import sqlite3
import numpy as np
import xml.etree.ElementTree as ET
import schedule
import threading
import html as html_lib
import webbrowser

from collections import deque, defaultdict
from datetime import datetime, timedelta

from db import create_attendance_table, get_connection


# =========================================================
# PROJECT PATHS AND SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPLOYEE_FOLDER = os.path.join(BASE_DIR, "employees")
TRAINING_FOLDER = os.path.join(EMPLOYEE_FOLDER, "training_photos")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODEL_FOLDER, "face_model.yml")
LABELS_PATH = os.path.join(MODEL_FOLDER, "labels.json")

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

CONFIDENCE_THRESHOLD = 85
STABLE_FRAMES_REQUIRED = 5
CAMERA_INDEX = 0

TAPO_RTSP_URL = os.environ.get("TAPO_RTSP_URL")

LINE_POSITION_RATIO = 0.50
LINE_MARGIN = 35

REPORT_TIME = "23:59"

os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)


# =========================================================
# FACE DETECTOR AND RECOGNIZER
# =========================================================

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise FileNotFoundError(
        "Haar cascade not found. Check the cascade file."
    )


def create_face_recognizer():

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Install OpenCV contrib using: "
            "pip install opencv-contrib-python"
        )

    return cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )


# =========================================================
# DATABASE HELPERS
# =========================================================

def get_employees():

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT employee_id, name, photo
            FROM employees
            ORDER BY employee_id
        """)

        return cursor.fetchall()

    finally:
        connection.close()


def get_employee_details(employee_id):

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT employee_id, name
            FROM employees
            WHERE employee_id = ?
        """, (str(employee_id),))

        return cursor.fetchone()

    finally:
        connection.close()


# =========================================================
# FACE PREPROCESSING
# =========================================================

def detect_faces(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    return gray, faces


def preprocess_face(face):

    face = cv2.resize(face, (200, 200))

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(face)


# =========================================================
# OPEN LAPTOP WEBCAM
# =========================================================

def open_camera():

    print("\nOpening laptop webcam...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Unable to open webcam.")
        camera.release()
        return None

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Webcam opened successfully.")
    print("Preparing webcam...")

    success = False
    frame = None

    for _ in range(30):
        success, frame = camera.read()
        time.sleep(0.1)

    if not success or frame is None:
        print("Unable to receive webcam frames.")
        camera.release()
        return None

    print("Camera brightness:", round(frame.mean(), 2))

    return camera


# =========================================================
# OPEN TAPO C500 CCTV
# =========================================================

def open_cctv_camera():

    print("\nConnecting to Tapo C500 CCTV...")

    rtsp_url = os.environ.get("TAPO_RTSP_URL") or TAPO_RTSP_URL

    if not rtsp_url:
        print("TAPO_RTSP_URL is not configured.")
        print("Set the environment variable before starting.")
        return None

    camera = cv2.VideoCapture(
        rtsp_url,
        cv2.CAP_FFMPEG
    )

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        camera.release()
        print("Unable to connect to Tapo CCTV.")
        print("Check the camera network and RTSP credentials.")
        return None

    print("Waiting for Tapo CCTV video...")

    for _ in range(30):

        success, frame = camera.read()

        if success and frame is not None:
            print("Tapo CCTV connected successfully!")
            print(
                "Video resolution:",
                frame.shape[1],
                "x",
                frame.shape[0]
            )
            return camera

        time.sleep(0.2)

    print("Unable to receive CCTV frames.")
    camera.release()
    return None


# =========================================================
# FACE AUGMENTATION
# =========================================================

def augment_face(face):

    brighter = cv2.convertScaleAbs(
        face,
        alpha=1.2,
        beta=20
    )

    darker = cv2.convertScaleAbs(
        face,
        alpha=0.8,
        beta=-10
    )

    flipped = cv2.flip(face, 1)

    return [
        face,
        brighter,
        darker,
        flipped
    ]


# =========================================================
# EMPLOYEE PHOTO PATHS
# =========================================================

def get_employee_photo_paths(employee_id, registered_photo):

    paths = []

    if registered_photo:

        if os.path.isabs(registered_photo):
            photo_path = registered_photo
        else:
            photo_path = os.path.join(
                BASE_DIR,
                registered_photo
            )

        if os.path.isfile(photo_path):
            paths.append(photo_path)

        else:
            alternative = os.path.join(
                EMPLOYEE_FOLDER,
                os.path.basename(registered_photo)
            )

            if os.path.isfile(alternative):
                paths.append(alternative)

    employee_folder = os.path.join(
        TRAINING_FOLDER,
        str(employee_id)
    )

    if os.path.isdir(employee_folder):

        for filename in os.listdir(employee_folder):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                paths.append(
                    os.path.join(employee_folder, filename)
                )

    return list(dict.fromkeys(paths))


# =========================================================
# CAPTURE TRAINING PHOTOS USING LAPTOP WEBCAM
# =========================================================

def capture_employee_photos():

    employees = get_employees()

    if not employees:
        print("No employees registered.")
        return

    print("\nREGISTERED EMPLOYEES")

    for employee_id, name, photo in employees:
        print(f"{employee_id} - {name}")

    employee_id = input(
        "\nEnter Employee ID: "
    ).strip()

    employee = get_employee_details(employee_id)

    if not employee:
        print("Employee ID not found.")
        return

    folder = os.path.join(
        TRAINING_FOLDER,
        str(employee_id)
    )

    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    saved_count = len(existing)

    camera = open_camera()

    if camera is None:
        return

    print("\nS = Save photo")
    print("Q = Finish")
    print("Capture 5-10 clear photos with different angles.")

    try:

        while True:

            success, frame = camera.read()

            if not success or frame is None:
                print("Unable to read webcam.")
                break

            display = frame.copy()

            gray, faces = detect_faces(frame)

            for (x, y, w, h) in faces:

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                display,
                "S: Save | Q: Finish",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Saved: {saved_count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Training Photo Capture",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):

                if len(faces) != 1:
                    print("Keep exactly one face in the frame.")
                    continue

                saved_count += 1

                filename = f"{employee_id}_{saved_count}.jpg"
                save_path = os.path.join(folder, filename)

                if cv2.imwrite(save_path, frame):
                    print("Saved:", save_path)
                else:
                    saved_count -= 1
                    print("Unable to save photo.")

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nPhoto capture completed.")
    print("Total photos in folder:", saved_count)


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    employees = get_employees()

    if not employees:
        print("No employees registered.")
        return

    faces_list = []
    labels_list = []
    label_mapping = {}

    current_label = 0

    print("\nSTARTING FACE MODEL TRAINING")

    for employee_id, name, registered_photo in employees:

        image_paths = get_employee_photo_paths(
            employee_id,
            registered_photo
        )

        employee_samples = 0
        real_photos_used = 0

        for image_path in image_paths:

            image = cv2.imread(image_path)

            if image is None:
                print("Unable to read:", image_path)
                continue

            gray, detected_faces = detect_faces(image)

            if len(detected_faces) == 0:
                print("No face detected:", image_path)
                continue

            x, y, w, h = max(
                detected_faces,
                key=lambda item: item[2] * item[3]
            )

            face = gray[y:y + h, x:x + w]
            face = preprocess_face(face)

            for sample in augment_face(face):
                faces_list.append(sample)
                labels_list.append(current_label)
                employee_samples += 1

            real_photos_used += 1

        if employee_samples > 0:

            label_mapping[str(current_label)] = {
                "employee_id": str(employee_id),
                "name": name
            }

            print(f"\nEmployee: {name}")
            print("Employee ID:", employee_id)
            print("Real photos used:", real_photos_used)
            print("Training samples:", employee_samples)

            current_label += 1

        else:
            print(f"No usable photos for {name}")

    if not faces_list:
        print("No training faces found.")
        return

    recognizer = create_face_recognizer()

    recognizer.train(
        faces_list,
        np.array(labels_list, dtype=np.int32)
    )

    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(label_mapping, file, indent=4)

    print("\nMODEL TRAINING COMPLETED")
    print("Employees trained:", current_label)
    print("Total training samples:", len(faces_list))
    print("Model saved:", MODEL_PATH)


# =========================================================
# LOAD MODEL
# =========================================================

def load_trained_model():

    if not os.path.isfile(MODEL_PATH):
        print("Model not found. Train the model first.")
        return None, None

    if not os.path.isfile(LABELS_PATH):
        print("Labels file not found. Train again.")
        return None, None

    recognizer = create_face_recognizer()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        label_mapping = json.load(file)

    return recognizer, label_mapping


# =========================================================
# DATABASE CHECK-IN
# =========================================================

def mark_checkin(employee_id, employee_name):

    create_attendance_table()

    connection = get_connection()

    try:

        cursor = connection.cursor()

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        cursor.execute("""
            SELECT id, check_in, check_out
            FROM attendance
            WHERE employee_id = ? AND date = ?
        """, (str(employee_id), today))

        record = cursor.fetchone()

        if record is None:

            cursor.execute("""
                INSERT INTO attendance
                (
                    employee_id,
                    employee_name,
                    date,
                    check_in,
                    check_out,
                    total_hours,
                    status
                )
                VALUES (?, ?, ?, ?, NULL, 0, 'Present')
            """, (
                str(employee_id),
                employee_name,
                today,
                current_time
            ))

            connection.commit()

            print(
                f"\nENTRY RECORDED: {employee_name} "
                f"at {current_time}"
            )

        else:

            print(
                f"\nAttendance record already exists for "
                f"{employee_name}. Check-in was not changed."
            )

    except sqlite3.Error as error:

        connection.rollback()
        print("Database error:", error)

    finally:
        connection.close()


# =========================================================
# DATABASE CHECK-OUT
# =========================================================

def mark_checkout(employee_id, employee_name):

    create_attendance_table()

    connection = get_connection()

    try:

        cursor = connection.cursor()

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        cursor.execute("""
            SELECT id, check_in, check_out
            FROM attendance
            WHERE employee_id = ? AND date = ?
        """, (str(employee_id), today))

        record = cursor.fetchone()

        if record is None:
            print(
                f"\nNo check-in found for {employee_name}. "
                "Exit not recorded."
            )
            return

        if record[2] is not None:
            print(
                f"\nCheck-out already exists for {employee_name}."
            )
            return

        if not record[1]:
            print("Check-in time is missing. Exit not recorded.")
            return

        check_in_datetime = datetime.strptime(
            today + " " + record[1],
            "%Y-%m-%d %H:%M:%S"
        )

        check_out_datetime = datetime.strptime(
            today + " " + current_time,
            "%Y-%m-%d %H:%M:%S"
        )

        if check_out_datetime < check_in_datetime:
            check_out_datetime += timedelta(days=1)

        total_seconds = (
            check_out_datetime - check_in_datetime
        ).total_seconds()

        total_hours = round(total_seconds / 3600, 2)

        cursor.execute("""
            UPDATE attendance
            SET check_out = ?, total_hours = ?
            WHERE id = ?
        """, (
            current_time,
            total_hours,
            record[0]
        ))

        connection.commit()

        print(
            f"\nEXIT RECORDED: {employee_name} "
            f"at {current_time}"
        )

        print("Total elapsed hours:", total_hours)

    except (sqlite3.Error, ValueError) as error:

        connection.rollback()
        print("Database error:", error)

    finally:
        connection.close()


# =========================================================
# REPORT HELPERS
# =========================================================

def format_duration(total_seconds):

    total_seconds = max(0, int(total_seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    return f"{hours}h {minutes}m"


def get_attendance_records(report_date=None):

    create_attendance_table()

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()

    try:

        cursor = connection.cursor()

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
            ORDER BY employee_id
        """, (report_date,))

        return cursor.fetchall()

    finally:
        connection.close()


def get_record_duration(record, generated_at):

    check_in = record[3]
    check_out = record[4]
    stored_hours = record[5]

    if not check_in:
        return "Not available", False

    if check_out:

        try:
            seconds = float(stored_hours or 0) * 3600
            return format_duration(seconds), False

        except (TypeError, ValueError):
            return "Not available", False

    try:

        start = datetime.strptime(
            record[2] + " " + check_in,
            "%Y-%m-%d %H:%M:%S"
        )

        end = generated_at

        if end < start:
            end += timedelta(days=1)

        elapsed = (end - start).total_seconds()

        return format_duration(elapsed) + " (In Progress)", True

    except ValueError:
        return "In Progress", True


# =========================================================
# PROFESSIONAL XML EXPORT
# =========================================================

def export_today_to_xml(records=None, report_date=None):

    generated_at = datetime.now()

    if report_date is None:
        report_date = generated_at.strftime("%Y-%m-%d")

    if records is None:

        try:
            records = get_attendance_records(report_date)

        except sqlite3.Error as error:
            print("Unable to read attendance for XML:", error)
            return None

    root = ET.Element("AttendanceReport")

    root.set("date", report_date)
    root.set(
        "generated_at",
        generated_at.strftime("%Y-%m-%d %H:%M:%S")
    )
    root.set("system", "CCTV Face Recognition Attendance System")

    summary = ET.SubElement(root, "Summary")

    total_employees = len(records)
    present_count = sum(
        1 for record in records
        if str(record[6]).lower() == "present"
    )
    active_count = sum(
        1 for record in records
        if record[3] and not record[4]
    )
    completed_count = sum(
        1 for record in records
        if record[3] and record[4]
    )

    ET.SubElement(
        summary, "TotalEmployees"
    ).text = str(total_employees)

    ET.SubElement(
        summary, "Present"
    ).text = str(present_count)

    ET.SubElement(
        summary, "CompletedSessions"
    ).text = str(completed_count)

    ET.SubElement(
        summary, "ActiveSessions"
    ).text = str(active_count)

    for record in records:

        employee = ET.SubElement(root, "Employee")

        fields = [
            "EmployeeID",
            "EmployeeName",
            "Date",
            "CheckIn",
            "CheckOut",
            "TotalHours",
            "Status"
        ]

        for field_name, value in zip(fields, record):

            element = ET.SubElement(employee, field_name)

            if value is None:
                element.text = ""
            else:
                element.text = str(value)

        ET.SubElement(
            employee,
            "AttendanceState"
        ).text = (
            "In Progress"
            if record[3] and not record[4]
            else "Completed"
        )

    tree = ET.ElementTree(root)

    ET.indent(tree, space="    ", level=0)

    filename = f"attendance_{report_date}.xml"
    xml_path = os.path.join(REPORTS_FOLDER, filename)

    try:

        tree.write(
            xml_path,
            encoding="utf-8",
            xml_declaration=True
        )

        print("\nXML ATTENDANCE REPORT EXPORTED")
        print("File:", xml_path)

        return xml_path

    except (OSError, ET.ParseError) as error:

        print("XML export failed:", error)
        return None


# =========================================================
# PROFESSIONAL HTML REPORT
# =========================================================

def export_today_to_html(records=None, report_date=None):

    generated_at = datetime.now()

    if report_date is None:
        report_date = generated_at.strftime("%Y-%m-%d")

    if records is None:

        try:
            records = get_attendance_records(report_date)

        except sqlite3.Error as error:
            print("Unable to read attendance for HTML:", error)
            return None

    total_employees = len(records)

    present_count = sum(
        1 for record in records
        if str(record[6]).lower() == "present"
    )

    active_count = sum(
        1 for record in records
        if record[3] and not record[4]
    )

    completed_count = sum(
        1 for record in records
        if record[3] and record[4]
    )

    rows = []

    for index, record in enumerate(records, start=1):

        employee_id = html_lib.escape(str(record[0]))
        employee_name = html_lib.escape(str(record[1]))
        employee_date = html_lib.escape(str(record[2]))
        check_in = html_lib.escape(str(record[3] or "—"))
        check_out = html_lib.escape(str(record[4] or "—"))
        status = html_lib.escape(str(record[6] or "Unknown"))

        duration, in_progress = get_record_duration(
            record,
            generated_at
        )

        duration = html_lib.escape(duration)

        if in_progress:
            state_html = (
                '<span class="status active">IN PROGRESS</span>'
            )
        elif record[3] and record[4]:
            state_html = (
                '<span class="status completed">COMPLETED</span>'
            )
        else:
            state_html = (
                '<span class="status pending">PENDING</span>'
            )

        rows.append(f"""
            <tr>
                <td>{index}</td>
                <td>{employee_id}</td>
                <td class="employee-name">{employee_name}</td>
                <td>{employee_date}</td>
                <td>{check_in}</td>
                <td>{check_out}</td>
                <td>{duration}</td>
                <td>{html_lib.escape(status)}</td>
                <td>{state_html}</td>
            </tr>
        """)

    if not rows:

        table_content = """
            <tr>
                <td colspan="9" class="empty">
                    No attendance records found for this date.
                </td>
            </tr>
        """

    else:
        table_content = "\n".join(rows)

    generated_text = generated_at.strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Employee Attendance Report</title>

<style>

:root {{
    --blue: #174ea6;
    --dark: #142b4a;
    --light-blue: #eaf2ff;
    --green: #16834a;
    --orange: #bd7100;
    --border: #dce4ef;
    --text: #26364b;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px;
    background: #f2f5fa;
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
}}

.container {{
    max-width: 1250px;
    margin: auto;
    background: white;
    padding: 35px;
    border-radius: 12px;
    box-shadow: 0 5px 25px rgba(20, 43, 74, 0.08);
}}

.header {{
    background: linear-gradient(120deg, #12366a, #2470c9);
    color: white;
    padding: 30px;
    border-radius: 10px;
    text-align: center;
}}

.header h1 {{
    margin: 0 0 10px;
    font-size: 28px;
    letter-spacing: 1px;
}}

.header p {{
    margin: 5px;
    opacity: 0.92;
}}

.report-date {{
    margin-top: 15px;
    font-weight: bold;
}}

.actions {{
    display: flex;
    justify-content: flex-end;
    margin: 22px 0;
    gap: 10px;
}}

button {{
    background: var(--blue);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px 20px;
    cursor: pointer;
    font-weight: bold;
    font-size: 14px;
}}

button:hover {{
    background: #103c80;
}}

.summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 25px 0 35px;
}}

.card {{
    padding: 22px;
    background: var(--light-blue);
    border-left: 5px solid var(--blue);
    border-radius: 8px;
}}

.card.present {{
    border-left-color: var(--green);
    background: #edf9f1;
}}

.card.active {{
    border-left-color: var(--orange);
    background: #fff6e8;
}}

.card.completed {{
    border-left-color: #54718f;
    background: #f0f4f8;
}}

.card .number {{
    display: block;
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 7px;
}}

.card .label {{
    font-size: 13px;
    color: #53647a;
    font-weight: bold;
    text-transform: uppercase;
}}

.section-title {{
    color: var(--dark);
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 18px;
}}

.table-wrapper {{
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

thead {{
    background: #173e70;
    color: white;
}}

th, td {{
    padding: 14px 11px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}}

tbody tr:nth-child(even) {{
    background: #f6f9fd;
}}

tbody tr:hover {{
    background: #edf4ff;
}}

.employee-name {{
    font-weight: bold;
    color: var(--dark);
}}

.status {{
    display: inline-block;
    padding: 6px 9px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: bold;
}}

.status.active {{
    color: #925600;
    background: #fff0ce;
}}

.status.completed {{
    color: #12643b;
    background: #dff5e7;
}}

.status.pending {{
    color: #53647a;
    background: #e8edf3;
}}

.empty {{
    text-align: center;
    padding: 35px;
    color: #718096;
}}

.note {{
    background: #f5f8fc;
    border-left: 4px solid var(--blue);
    padding: 14px;
    margin-top: 25px;
    font-size: 12px;
    line-height: 1.7;
}}

.footer {{
    text-align: center;
    color: #718096;
    font-size: 11px;
    margin-top: 30px;
    border-top: 1px solid var(--border);
    padding-top: 18px;
}}

@media (max-width: 750px) {{
    body {{
        padding: 10px;
    }}

    .container {{
        padding: 15px;
    }}

    .summary {{
        grid-template-columns: repeat(2, 1fr);
    }}

    .header h1 {{
        font-size: 21px;
    }}
}}

@media print {{
    @page {{
        size: landscape;
        margin: 12mm;
    }}

    body {{
        background: white;
        padding: 0;
    }}

    .container {{
        max-width: none;
        padding: 0;
        box-shadow: none;
    }}

    .actions {{
        display: none;
    }}

    .header {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}

    thead {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}

    .card {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}

    tr {{
        page-break-inside: avoid;
    }}

    .status {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}
}}

</style>
</head>

<body>

<div class="container">

    <header class="header">
        <h1>EMPLOYEE ATTENDANCE REPORT</h1>
        <p>CCTV FACE RECOGNITION ATTENDANCE MANAGEMENT SYSTEM</p>
        <p class="report-date">Report Date: {html_lib.escape(report_date)}</p>
        <p>Generated: {html_lib.escape(generated_text)}</p>
    </header>

    <div class="actions">
        <button onclick="window.print()">
            Print / Save as PDF
        </button>
    </div>

    <section class="summary">

        <div class="card">
            <span class="number">{total_employees}</span>
            <span class="label">Total Employees</span>
        </div>

        <div class="card present">
            <span class="number">{present_count}</span>
            <span class="label">Present</span>
        </div>

        <div class="card active">
            <span class="number">{active_count}</span>
            <span class="label">Active Sessions</span>
        </div>

        <div class="card completed">
            <span class="number">{completed_count}</span>
            <span class="label">Completed Sessions</span>
        </div>

    </section>

    <h2 class="section-title">Attendance Details</h2>

    <div class="table-wrapper">

        <table>

            <thead>
                <tr>
                    <th>S.No.</th>
                    <th>Employee ID</th>
                    <th>Employee Name</th>
                    <th>Date</th>
                    <th>Check In</th>
                    <th>Check Out</th>
                    <th>Working Duration</th>
                    <th>Attendance</th>
                    <th>Session</th>
                </tr>
            </thead>

            <tbody>
                {table_content}
            </tbody>

        </table>

    </div>

    <div class="note">
        <strong>Report Notes:</strong><br>
        • In Progress means the employee has checked in but has not
        checked out yet.<br>
        • Working duration for an active session is provisional and
        calculated up to the report generation time.<br>
        • Recorded working duration is based on elapsed time between
        check-in and check-out. It does not deduct breaks.
    </div>

    <footer class="footer">
        Generated by CCTV Face Recognition Attendance Management System
        <br>
        This is a computer-generated attendance report.
    </footer>

</div>

</body>
</html>
"""

    filename = f"attendance_{report_date}.html"
    html_path = os.path.join(REPORTS_FOLDER, filename)

    try:

        with open(
            html_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report_html)

        print("\nPROFESSIONAL HTML REPORT EXPORTED")
        print("File:", html_path)

        return html_path

    except OSError as error:

        print("HTML export failed:", error)
        return None


# =========================================================
# EXPORT BOTH XML AND HTML REPORTS
# =========================================================

def export_today_reports(open_html=False):

    report_date = datetime.now().strftime("%Y-%m-%d")

    try:

        records = get_attendance_records(report_date)

    except sqlite3.Error as error:

        print("Unable to retrieve attendance:", error)
        return None, None

    print("\nGENERATING ATTENDANCE REPORTS...")
    print("--------------------------------")

    xml_path = export_today_to_xml(
        records,
        report_date
    )

    html_path = export_today_to_html(
        records,
        report_date
    )

    print("\nREPORT GENERATION COMPLETED")
    print("XML:", xml_path)
    print("HTML:", html_path)

    if open_html and html_path:

        try:
            webbrowser.open(
                "file:///" + html_path.replace("\\", "/")
            )
            print("Opening HTML report in your browser...")

        except Exception as error:
            print("Unable to open browser:", error)

    return xml_path, html_path


# =========================================================
# AUTOMATIC DAILY REPORT EXPORT
# =========================================================

def automatic_daily_report_worker():

    while True:

        try:
            schedule.run_pending()

        except Exception as error:
            print("Automatic report error:", error)

        time.sleep(30)


def start_automatic_report_export():

    schedule.clear("daily_attendance_report")

    schedule.every().day.at(REPORT_TIME).do(
        export_today_reports
    ).tag("daily_attendance_report")

    report_thread = threading.Thread(
        target=automatic_daily_report_worker,
        daemon=True
    )

    report_thread.start()

    print("\nAUTOMATIC DAILY REPORT EXPORT ENABLED")
    print("Daily export time:", REPORT_TIME)
    print("Reports will be saved in:", REPORTS_FOLDER)
    print("Keep the application running until the scheduled time.")


# =========================================================
# IMAGE RECOGNITION - NO ATTENDANCE CHANGE
# =========================================================

def recognize_employee(image_path=None):

    recognizer, label_mapping = load_trained_model()

    if recognizer is None:
        return

    if image_path is None:
        image_path = input(
            "\nEnter employee image path: "
        ).strip().strip('"')

    image = cv2.imread(image_path)

    if image is None:
        print("Unable to open image.")
        return

    gray, faces = detect_faces(image)

    if len(faces) == 0:
        print("No face detected.")
        return

    for (x, y, w, h) in faces:

        face = preprocess_face(
            gray[y:y + h, x:x + w]
        )

        label, confidence = recognizer.predict(face)

        if (
            str(label) in label_mapping
            and confidence <= CONFIDENCE_THRESHOLD
        ):

            employee = label_mapping[str(label)]

            text = (
                f"{employee['name']} "
                f"({employee['employee_id']})"
            )

            color = (0, 255, 0)

            print("\nEMPLOYEE RECOGNIZED:", text)

        else:

            text = "Unknown Person"
            color = (0, 0, 255)

            print("\nUNKNOWN PERSON")

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            image,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    print(
        "\nImage recognition only. "
        "Attendance is not changed by this option."
    )

    cv2.imshow("Employee Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================================================
# LIVE ENTRY / EXIT DETECTION USING TAPO CCTV
# =========================================================

def live_webcam_recognition():

    recognizer, label_mapping = load_trained_model()

    if recognizer is None:
        return

    camera = open_cctv_camera()

    if camera is None:
        return

    prediction_history = defaultdict(
        lambda: deque(maxlen=STABLE_FRAMES_REQUIRED)
    )

    employee_sides = {}

    print("\nLIVE TAPO CCTV ENTRY / EXIT DETECTION STARTED")
    print("----------------------------------------------")
    print("LEFT TO RIGHT  = ENTRY / CHECK-IN")
    print("RIGHT TO LEFT = EXIT / CHECK-OUT")
    print("Press Q to stop.")
    print("----------------------------------------------")

    try:

        while True:

            success, frame = camera.read()

            if not success or frame is None:
                print("Unable to read CCTV frame. Stopping monitoring.")
                break

            height, width = frame.shape[:2]

            line_x = int(width * LINE_POSITION_RATIO)

            gray, faces = detect_faces(frame)

            cv2.line(
                frame,
                (line_x, 0),
                (line_x, height),
                (255, 255, 0),
                2
            )

            cv2.line(
                frame,
                (line_x - LINE_MARGIN, 0),
                (line_x - LINE_MARGIN, height),
                (100, 100, 100),
                1
            )

            cv2.line(
                frame,
                (line_x + LINE_MARGIN, 0),
                (line_x + LINE_MARGIN, height),
                (100, 100, 100),
                1
            )

            cv2.putText(
                frame,
                "ENTRY: LEFT -> RIGHT",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "EXIT: RIGHT -> LEFT",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

            for (x, y, w, h) in faces:

                face = preprocess_face(
                    gray[y:y + h, x:x + w]
                )

                label, confidence = recognizer.predict(face)

                if (
                    str(label) not in label_mapping
                    or confidence > CONFIDENCE_THRESHOLD
                ):

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Unknown Person",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    continue

                employee = label_mapping[str(label)]

                employee_id = employee["employee_id"]
                employee_name = employee["name"]

                prediction_history[employee_id].append(label)

                history = prediction_history[employee_id]

                stable = (
                    len(history) == STABLE_FRAMES_REQUIRED
                    and len(set(history)) == 1
                )

                center_x = x + w // 2

                if not stable:

                    display_text = "Verifying..."
                    color = (0, 255, 255)

                else:

                    display_text = (
                        f"{employee_name} ({employee_id})"
                    )

                    color = (0, 255, 0)

                    if center_x < line_x - LINE_MARGIN:
                        current_side = "LEFT"

                    elif center_x > line_x + LINE_MARGIN:
                        current_side = "RIGHT"

                    else:
                        current_side = None

                    previous_side = employee_sides.get(employee_id)

                    if previous_side is None:

                        if current_side is not None:
                            employee_sides[employee_id] = current_side

                    elif current_side is not None:

                        if current_side != previous_side:

                            if (
                                previous_side == "LEFT"
                                and current_side == "RIGHT"
                            ):

                                print(
                                    f"\nENTRY CROSSING: {employee_name}"
                                )

                                mark_checkin(
                                    employee_id,
                                    employee_name
                                )

                                display_text = "ENTRY DETECTED"

                            elif (
                                previous_side == "RIGHT"
                                and current_side == "LEFT"
                            ):

                                print(
                                    f"\nEXIT CROSSING: {employee_name}"
                                )

                                mark_checkout(
                                    employee_id,
                                    employee_name
                                )

                                display_text = "EXIT DETECTED"

                            employee_sides[employee_id] = current_side

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, y + h // 2),
                    5,
                    (255, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    display_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.putText(
                frame,
                "Q = STOP",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "Tapo CCTV Entry Exit Attendance",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQ pressed. Stopping CCTV monitoring.")
                break

    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")

    except Exception as error:
        print("\nCCTV monitoring error:", error)

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("\nTapo CCTV monitoring stopped.")

        # Export both reports after monitoring stops.
        export_today_reports()


# =========================================================
# MAIN MENU
# =========================================================

def main():

    create_attendance_table()

    # Start automatic XML and HTML report scheduler.
    start_automatic_report_export()

    while True:

        print("\n============================================")
        print(" CCTV FACE RECOGNITION ATTENDANCE SYSTEM")
        print("============================================")

        print("1. Train Face Recognition Model")
        print("2. Recognize Employee from Image")
        print("3. Live Tapo CCTV Entry / Exit Detection")
        print("4. Capture Employee Training Photos")
        print("5. Export Today's XML + HTML Reports")
        print("6. Open Today's HTML Attendance Report")
        print("7. Exit")

        choice = input(
            "Enter your choice (1/2/3/4/5/6/7): "
        ).strip()

        if choice == "1":

            train_model()

        elif choice == "2":

            recognize_employee()

        elif choice == "3":

            live_webcam_recognition()

        elif choice == "4":

            capture_employee_photos()

        elif choice == "5":

            export_today_reports()

        elif choice == "6":

            xml_path, html_path = export_today_reports(
                open_html=True
            )

        elif choice == "7":

            print("\nGenerating final attendance reports...")

            export_today_reports()

            print("\nExiting CCTV Attendance System.")
            break

        else:

            print("\nInvalid choice. Try again.")


# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram interrupted.")

        # Attempt to save reports on keyboard interruption.
        try:
            export_today_reports()
        except Exception as error:
            print("Final report export error:", error)
