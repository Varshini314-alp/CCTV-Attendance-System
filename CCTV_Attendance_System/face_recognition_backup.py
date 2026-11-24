
import cv2
import os
import json
import time
import sqlite3
import numpy as np

from collections import deque, defaultdict
from datetime import datetime

from db import create_attendance_table, get_connection


# =========================================================
# PROJECT PATHS AND SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPLOYEE_FOLDER = os.path.join(BASE_DIR, "employees")
TRAINING_FOLDER = os.path.join(EMPLOYEE_FOLDER, "training_photos")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_FOLDER, "face_model.yml")
LABELS_PATH = os.path.join(MODEL_FOLDER, "labels.json")

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

CONFIDENCE_THRESHOLD = 85
STABLE_FRAMES_REQUIRED = 5
CAMERA_INDEX = 0

# Virtual line settings
LINE_POSITION_RATIO = 0.50
LINE_MARGIN = 35

# Forget a previous face position after this many seconds
TRACK_TIMEOUT = 2.0

os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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
# OPEN WEBCAM
# =========================================================

def open_camera():

    print("\nOpening webcam...")

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

    print("Camera opened successfully.")
    print("Preparing webcam...")

    success = False
    frame = None

    for _ in range(30):

        success, frame = camera.read()
        time.sleep(0.1)

    if not success or frame is None:

        print("Unable to receive camera frames.")
        camera.release()
        return None

    print("Camera brightness:", round(frame.mean(), 2))

    return camera


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
# CAPTURE TRAINING PHOTOS
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

    # Avoid overwriting existing training photos.
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

    while True:

        success, frame = camera.read()

        if not success:
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

    with open(LABELS_PATH, "w") as file:
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

    with open(LABELS_PATH, "r") as file:
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
                f"\nCheck-in already exists for {employee_name}."
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

        check_in_datetime = datetime.strptime(
            today + " " + record[1],
            "%Y-%m-%d %H:%M:%S"
        )

        check_out_datetime = datetime.strptime(
            today + " " + current_time,
            "%Y-%m-%d %H:%M:%S"
        )

        if check_out_datetime < check_in_datetime:
            from datetime import timedelta
            check_out_datetime += timedelta(days=1)

        total_seconds = (
            check_out_datetime - check_in_datetime
        ).total_seconds()

        total_hours = round(
            total_seconds / 3600,
            2
        )

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

        print("Total working hours:", total_hours)

    except sqlite3.Error as error:

        connection.rollback()
        print("Database error:", error)

    finally:
        connection.close()


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
# LIVE ENTRY AND EXIT DETECTION
# =========================================================

def live_webcam_recognition():

    recognizer, label_mapping = load_trained_model()

    if recognizer is None:
        return

    camera = open_camera()

    if camera is None:
        return

    prediction_history = defaultdict(
        lambda: deque(maxlen=STABLE_FRAMES_REQUIRED)
    )

    # Track the previous side of the line for each employee.
    employee_sides = {}

    # Track the last time each employee was recognized.
    employee_last_seen = {}

    # Track the last attendance action to prevent rapid repeats.
    employee_last_action = {}

    print("\nLIVE ENTRY / EXIT DETECTION STARTED")
    print("--------------------------------------")
    print("LEFT TO RIGHT  = ENTRY / CHECK-IN")
    print("RIGHT TO LEFT  = EXIT / CHECK-OUT")
    print("Press Q to stop.")
    print("--------------------------------------")

    while True:

        success, frame = camera.read()

        if not success:
            print("Unable to read webcam.")
            break

        height, width = frame.shape[:2]

        line_x = int(width * LINE_POSITION_RATIO)

        gray, faces = detect_faces(frame)

        current_time = time.time()

        # Draw virtual line and direction labels.
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

        for index, (x, y, w, h) in enumerate(faces):

            face = preprocess_face(
                gray[y:y + h, x:x + w]
            )

            label, confidence = recognizer.predict(face)

            if (
                str(label) not in label_mapping
                or confidence > CONFIDENCE_THRESHOLD
            ):

                prediction_history[index].clear()

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

                employee_last_seen[employee_id] = current_time

                # Determine which side of the virtual line
                # the employee is currently on.
                if center_x < line_x - LINE_MARGIN:

                    current_side = "LEFT"

                elif center_x > line_x + LINE_MARGIN:

                    current_side = "RIGHT"

                else:

                    current_side = None

                previous_side = employee_sides.get(
                    employee_id
                )

                # If the employee has disappeared for too long,
                # reset the previous position.
                last_seen = employee_last_seen.get(
                    employee_id
                )

                # First reliable observation establishes
                # the starting side. It does not mark attendance.
                if previous_side is None:

                    if current_side is not None:
                        employee_sides[employee_id] = current_side

                elif current_side is not None:

                    if current_side != previous_side:

                        # LEFT -> RIGHT means ENTRY.
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

                            employee_last_action[employee_id] = (
                                current_time
                            )

                            display_text = "ENTRY DETECTED"

                        # RIGHT -> LEFT means EXIT.
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

                            employee_last_action[employee_id] = (
                                current_time
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
            "CCTV Entry Exit Attendance",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\nLive webcam stopped.")


# =========================================================
# MAIN MENU
# =========================================================

def main():

    create_attendance_table()

    while True:

        print("\n================================")
        print(" CCTV FACE RECOGNITION SYSTEM")
        print("================================")

        print("1. Train Face Recognition Model")
        print("2. Recognize Employee from Image")
        print("3. Live Entry / Exit Detection")
        print("4. Capture Employee Training Photos")
        print("5. Exit")

        choice = input(
            "Enter your choice (1/2/3/4/5): "
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
            print("\nExiting CCTV Attendance System.")
            break

        else:
            print("\nInvalid choice. Try again.")


# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":
    main()
