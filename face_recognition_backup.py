
import os
import cv2
import json
import time
import threading
import numpy as np

from datetime import datetime, date
from collections import defaultdict

from db import create_attendance_table, get_connection
from reports import generate_daily_xml


# =========================================================
# 1. PROJECT SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPLOYEE_DIR = os.path.join(BASE_DIR, "employees")
TRAINING_DIR = os.path.join(EMPLOYEE_DIR, "training_photos")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

OFFICE_OPEN = "09:00"
OFFICE_CLOSE = "18:00"

CONFIDENCE_THRESHOLD = 85.0
STABLE_FRAMES_REQUIRED = 5

LINE_POSITION_RATIO = 0.50
LINE_MARGIN = 35
FRAME_WIDTH = 960

# LEFT -> RIGHT = ENTRY
# RIGHT -> LEFT = EXIT
ENTRY_DIRECTION = "LEFT_TO_RIGHT"

SIDE_STABLE_FRAMES = 4
CROSSING_COOLDOWN_SECONDS = 4
STALE_TRACK_SECONDS = 8
RECONNECT_DELAY_SECONDS = 3

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMPLOYEE_DIR, exist_ok=True)


# =========================================================
# 2. CAMERA VARIABLES
# =========================================================

FRAME_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()

LATEST_JPEG = None

CAMERA_STOP_EVENT = threading.Event()
CAMERA_THREAD = None

CAMERA_STATUS = "STOPPED"
CAMERA_ERROR = ""


def set_camera_status(status, error=""):
    global CAMERA_STATUS, CAMERA_ERROR

    with STATUS_LOCK:
        CAMERA_STATUS = status
        CAMERA_ERROR = str(error) if error else ""


def get_camera_status():
    with STATUS_LOCK:
        return {
            "status": CAMERA_STATUS,
            "error": CAMERA_ERROR
        }


# =========================================================
# 3. DATABASE
# =========================================================

def ensure_tables():
    create_attendance_table()


def get_employee(employee_id):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT employee_id, name
            FROM employees
            WHERE employee_id = ?
        """, (str(employee_id),))

        row = cursor.fetchone()

        if row:
            return str(row[0]), row[1]

        return None, None

    finally:
        conn.close()


def get_latest_attendance(employee_id, work_date=None):

    if work_date is None:
        work_date = date.today().isoformat()

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, check_in, check_out
            FROM attendance
            WHERE employee_id = ?
              AND date = ?
            ORDER BY id DESC
            LIMIT 1
        """, (str(employee_id), work_date))

        return cursor.fetchone()

    finally:
        conn.close()


def get_live_attendance_state(employee_id):

    row = get_latest_attendance(employee_id)

    if row is None:
        return "UNKNOWN"

    if row[2] is None:
        return "IN"

    return "LOGGED_OUT"


# =========================================================
# 4. XML REPORT
# =========================================================

def update_xml_report(work_date=None):

    try:
        path = generate_daily_xml(work_date)
        print("[XML] Report updated:", path)

    except Exception as error:
        print("[XML ERROR]", error)


# =========================================================
# 5. CHECK-IN
# =========================================================

def mark_checkin(employee_id, employee_name, now=None):

    if now is None:
        now = datetime.now()

    work_date = now.date().isoformat()
    login_time = now.strftime("%H:%M:%S")

    if now.strftime("%H:%M") >= OFFICE_CLOSE:
        print("[INFO] Check-in rejected after closing:", employee_name)
        return False

    latest = get_latest_attendance(employee_id, work_date)

    if latest is not None and latest[2] is None:
        print("[INFO] Already checked in:", employee_name)
        return False

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO attendance (
                employee_id,
                employee_name,
                date,
                check_in,
                check_out,
                total_hours,
                status,
                camera_id,
                attendance_source
            )
            VALUES (?, ?, ?, ?, NULL, 0, 'Present',
                    'TAPO_C500', 'CCTV')
        """, (
            str(employee_id),
            employee_name,
            work_date,
            login_time
        ))

        conn.commit()

        print(
            f"[CHECK-IN] {employee_name} "
            f"({employee_id}) at {login_time}"
        )

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    update_xml_report(work_date)
    return True


# =========================================================
# 6. CHECK-OUT
# =========================================================

def mark_checkout(employee_id, employee_name, now=None):

    if now is None:
        now = datetime.now()

    work_date = now.date().isoformat()
    logout_time = now.strftime("%H:%M:%S")

    latest = get_latest_attendance(employee_id, work_date)

    if latest is None:
        print("[WARNING] No check-in found:", employee_name)
        return False

    attendance_id, login_time, existing_logout = latest

    if existing_logout is not None:
        print("[INFO] Already checked out:", employee_name)
        return False

    login_datetime = datetime.strptime(
        f"{work_date} {login_time}",
        "%Y-%m-%d %H:%M:%S"
    )

    if now < login_datetime:
        print("[WARNING] Checkout earlier than check-in.")
        return False

    total_hours = round(
        (now - login_datetime).total_seconds() / 3600,
        2
    )

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE attendance
            SET check_out = ?,
                total_hours = ?,
                status = 'Present'
            WHERE id = ?
              AND check_out IS NULL
        """, (
            logout_time,
            total_hours,
            attendance_id
        ))

        updated = cursor.rowcount
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    if updated == 0:
        return False

    print(
        f"[CHECK-OUT] {employee_name} "
        f"({employee_id}) at {logout_time}"
    )

    print("[WORKING HOURS]", total_hours)

    update_xml_report(work_date)
    return True


# =========================================================
# 7. AUTOMATIC OFFICE CLOSING
# =========================================================

def finalize_open_attendance_at_closing(now=None):

    if now is None:
        now = datetime.now()

    if now.strftime("%H:%M") < OFFICE_CLOSE:
        return

    work_date = now.date().isoformat()

    closing_datetime = datetime.combine(
        now.date(),
        datetime.strptime(OFFICE_CLOSE, "%H:%M").time()
    )

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT employee_id, employee_name
            FROM attendance
            WHERE date = ?
              AND check_in IS NOT NULL
              AND check_out IS NULL
        """, (work_date,))

        employees = cursor.fetchall()

    finally:
        conn.close()

    for employee_id, employee_name in employees:
        mark_checkout(
            employee_id,
            employee_name,
            closing_datetime
        )


# =========================================================
# 8. HAAR CASCADE
# =========================================================

def load_cascade():

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if cascade.empty():
        raise FileNotFoundError(
            "Haar cascade missing: " + CASCADE_PATH
        )

    return cascade


# =========================================================
# 9. LOAD LBPH MODEL
# =========================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Face model missing. Train the model first."
        )

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "labels.json missing. Train the model first."
        )

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module unavailable. "
            "Install opencv-contrib-python."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    label_to_employee = {
        int(key): str(value)
        for key, value in labels.items()
    }

    return recognizer, label_to_employee


# =========================================================
# 10. TRAINING IMAGE COLLECTION
# =========================================================

def get_face_images():

    images = []
    employee_ids = []

    if os.path.isdir(EMPLOYEE_DIR):

        for filename in os.listdir(EMPLOYEE_DIR):

            path = os.path.join(EMPLOYEE_DIR, filename)

            if not os.path.isfile(path):
                continue

            employee_id = filename.split("_", 1)[0]

            if employee_id.isdigit():
                images.append(path)
                employee_ids.append(employee_id)

    if os.path.isdir(TRAINING_DIR):

        for employee_id in os.listdir(TRAINING_DIR):

            folder = os.path.join(TRAINING_DIR, employee_id)

            if not os.path.isdir(folder):
                continue

            for filename in os.listdir(folder):

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):
                    images.append(os.path.join(folder, filename))
                    employee_ids.append(str(employee_id))

    return images, employee_ids


# =========================================================
# 11. TRAIN MODEL
# =========================================================

def train_model():

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Install opencv-contrib-python."
        )

    image_paths, employee_ids = get_face_images()

    if not image_paths:
        print("[TRAINING] No training photos found.")
        return

    cascade = load_cascade()

    unique_ids = sorted(set(employee_ids))

    id_to_label = {
        employee_id: index
        for index, employee_id in enumerate(unique_ids)
    }

    label_to_id = {
        str(index): employee_id
        for employee_id, index in id_to_label.items()
    }

    faces = []
    labels = []
    valid_images = 0

    for path, employee_id in zip(image_paths, employee_ids):

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(detected) == 0:
            continue

        x, y, w, h = max(
            detected,
            key=lambda box: box[2] * box[3]
        )

        face = cv2.resize(
            gray[y:y+h, x:x+w],
            (200, 200)
        )

        label = id_to_label[employee_id]

        augmented_faces = [
            face,
            cv2.convertScaleAbs(face, alpha=1.15, beta=12),
            cv2.convertScaleAbs(face, alpha=0.85, beta=-10),
            cv2.flip(face, 1)
        ]

        faces.extend(augmented_faces)
        labels.extend([label] * len(augmented_faces))

        valid_images += 1

    if not faces:
        print("[TRAINING] No valid faces detected.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        faces,
        np.array(labels, dtype=np.int32)
    )

    recognizer.write(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(label_to_id, file, indent=4)

    print("\n================================")
    print("FACE MODEL TRAINING COMPLETED")
    print("================================")
    print("Employees:", len(unique_ids))
    print("Valid images:", valid_images)
    print("Training samples:", len(faces))


# =========================================================
# 12. TAPO RTSP CONNECTION
# =========================================================

def open_tapo_camera():

    # Read environment variable at camera-start time.
    # This avoids using an outdated value loaded at import.
    rtsp_url = os.environ.get(
        "TAPO_RTSP_URL", ""
    ).strip()

    if not rtsp_url:
        raise RuntimeError(
            "TAPO_RTSP_URL is missing. "
            "Set it in the PowerShell window "
            "before starting Flask."
        )

    print("[CAMERA] Connecting to Tapo C500...")

    cap = cv2.VideoCapture(
        rtsp_url,
        cv2.CAP_FFMPEG
    )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not open RTSP stream. "
            "Check camera IP, account credentials, "
            "stream path and network."
        )

    print("[CAMERA] RTSP connection opened.")

    return cap


# =========================================================
# 13. FACE RECOGNITION
# =========================================================

def recognize_face(
    gray_face,
    recognizer,
    label_to_employee
):

    resized_face = cv2.resize(
        gray_face,
        (200, 200)
    )

    label, confidence = recognizer.predict(resized_face)

    if confidence > CONFIDENCE_THRESHOLD:
        return None, confidence

    employee_id = label_to_employee.get(int(label))

    if not employee_id:
        return None, confidence

    employee_id, employee_name = get_employee(employee_id)

    if not employee_id:
        return None, confidence

    return (employee_id, employee_name), confidence


# =========================================================
# 14. PUBLISH VIDEO FRAME
# =========================================================

def publish_frame(frame):

    global LATEST_JPEG

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )

    if success:
        with FRAME_LOCK:
            LATEST_JPEG = encoded.tobytes()


def generate_video_frames():

    while True:

        with FRAME_LOCK:
            frame_bytes = LATEST_JPEG

        if frame_bytes is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.04)


# =========================================================
# 15. LIVE CCTV RECOGNITION
# =========================================================

def live_cctv_recognition(
    show_window=False,
    stop_event=None
):

    global LATEST_JPEG

    if stop_event is None:
        stop_event = CAMERA_STOP_EVENT

    ensure_tables()

    cascade = load_cascade()
    recognizer, label_to_employee = load_model()

    if ENTRY_DIRECTION == "LEFT_TO_RIGHT":
        entry_side = "RIGHT"
        exit_side = "LEFT"
    else:
        entry_side = "LEFT"
        exit_side = "RIGHT"

    identity_counts = defaultdict(int)

    confirmed_side = {}
    candidate_side = {}
    candidate_count = defaultdict(int)

    last_event_time = defaultdict(float)
    last_seen = {}

    employee_state = {}

    cap = None

    with FRAME_LOCK:
        LATEST_JPEG = None

    print("\n========================================")
    print("TAPO C500 LIVE ATTENDANCE STARTED")
    print("========================================")

    print("Entry direction:", ENTRY_DIRECTION)
    print("Office hours:", OFFICE_OPEN, "-", OFFICE_CLOSE)

    try:

        while not stop_event.is_set():

            # ---------------------------------------------
            # CONNECT OR RECONNECT
            # ---------------------------------------------

            if cap is None:

                set_camera_status("CONNECTING")

                try:
                    cap = open_tapo_camera()

                except Exception as error:

                    set_camera_status("RECONNECTING", error)

                    print("[CAMERA CONNECTION ERROR]", error)

                    if stop_event.wait(RECONNECT_DELAY_SECONDS):
                        break

                    continue

            # ---------------------------------------------
            # READ FRAME
            # ---------------------------------------------

            ok, frame = cap.read()

            if not ok or frame is None:

                error = "Unable to read RTSP frame. Reconnecting."

                print("[CAMERA]", error)

                set_camera_status("RECONNECTING", error)

                cap.release()
                cap = None

                if stop_event.wait(RECONNECT_DELAY_SECONDS):
                    break

                continue

            set_camera_status("LIVE")

            height, width = frame.shape[:2]

            if width > FRAME_WIDTH:

                scale = FRAME_WIDTH / width

                frame = cv2.resize(
                    frame,
                    (
                        FRAME_WIDTH,
                        int(height * scale)
                    )
                )

                height, width = frame.shape[:2]

            line_x = int(width * LINE_POSITION_RATIO)

            cv2.line(
                frame,
                (line_x, 0),
                (line_x, height),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "ENTRY >>>",
                (max(10, line_x - 150), 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(65, 65)
            )

            now_mono = time.monotonic()
            now_dt = datetime.now()

            seen_this_frame = set()

            # ---------------------------------------------
            # PROCESS FACES
            # ---------------------------------------------

            for (x, y, w, h) in detected_faces:

                face_gray = gray[y:y+h, x:x+w]

                identity, confidence = recognize_face(
                    face_gray,
                    recognizer,
                    label_to_employee
                )

                if identity is None:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Unknown",
                        (x, max(20, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    continue

                employee_id, employee_name = identity

                seen_this_frame.add(employee_id)

                identity_counts[employee_id] += 1
                last_seen[employee_id] = now_mono

                center_x = x + w // 2
                center_y = y + h // 2

                if center_x < line_x - LINE_MARGIN:
                    side = "LEFT"
                elif center_x > line_x + LINE_MARGIN:
                    side = "RIGHT"
                else:
                    side = "MIDDLE"

                # -----------------------------------------
                # STABLE FACE VERIFICATION
                # -----------------------------------------

                if (
                    identity_counts[employee_id]
                    >= STABLE_FRAMES_REQUIRED
                    and side in ("LEFT", "RIGHT")
                ):

                    if employee_id not in employee_state:

                        employee_state[employee_id] = (
                            get_live_attendance_state(employee_id)
                        )

                    if employee_id not in confirmed_side:

                        confirmed_side[employee_id] = side

                    else:

                        if side != confirmed_side[employee_id]:

                            if side != candidate_side.get(employee_id):

                                candidate_side[employee_id] = side
                                candidate_count[employee_id] = 1

                            else:
                                candidate_count[employee_id] += 1

                        else:

                            candidate_side.pop(employee_id, None)
                            candidate_count[employee_id] = 0

                        crossed = (
                            side != confirmed_side[employee_id]
                            and candidate_count[employee_id]
                            >= SIDE_STABLE_FRAMES
                            and (
                                now_mono - last_event_time[employee_id]
                            ) >= CROSSING_COOLDOWN_SECONDS
                        )

                        if crossed:

                            previous_side = confirmed_side[employee_id]

                            crossed_into_office = (
                                previous_side != entry_side
                                and side == entry_side
                            )

                            crossed_out_of_office = (
                                previous_side == entry_side
                                and side == exit_side
                            )

                            if crossed_into_office:

                                if now_dt.strftime("%H:%M") >= OFFICE_CLOSE:

                                    print(
                                        "[INFO] Entry ignored after closing:",
                                        employee_name
                                    )

                                elif employee_state[employee_id] in (
                                    "UNKNOWN",
                                    "LOGGED_OUT"
                                ):

                                    if mark_checkin(
                                        employee_id,
                                        employee_name,
                                        now_dt
                                    ):
                                        employee_state[employee_id] = "IN"

                                else:

                                    print(
                                        "[INFO] Already checked in:",
                                        employee_name
                                    )

                                print(
                                    "[ENTRY CROSSING]",
                                    employee_name,
                                    employee_id
                                )

                            elif crossed_out_of_office:

                                if employee_state[employee_id] == "IN":

                                    if mark_checkout(
                                        employee_id,
                                        employee_name,
                                        now_dt
                                    ):
                                        employee_state[employee_id] = "LOGGED_OUT"

                                    print(
                                        "[EXIT CROSSING]",
                                        employee_name,
                                        employee_id
                                    )

                                else:

                                    print(
                                        "[INFO] No active check-in:",
                                        employee_name
                                    )

                            confirmed_side[employee_id] = side
                            last_event_time[employee_id] = now_mono

                            candidate_side.pop(employee_id, None)
                            candidate_count[employee_id] = 0

                # -----------------------------------------
                # DISPLAY STATUS
                # -----------------------------------------

                display_state = employee_state.get(
                    employee_id,
                    "VERIFYING"
                )

                if display_state == "IN":
                    color = (0, 200, 0)
                elif display_state == "LOGGED_OUT":
                    color = (255, 180, 0)
                else:
                    color = (0, 200, 255)

                label_text = (
                    f"{employee_name} | "
                    f"{employee_id} | "
                    f"{display_state}"
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label_text,
                    (x, max(22, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (255, 0, 255),
                    -1
                )

            # ---------------------------------------------
            # RESET FACE COUNTERS
            # ---------------------------------------------

            for employee_id in list(identity_counts):

                if employee_id not in seen_this_frame:
                    identity_counts[employee_id] = 0

            # ---------------------------------------------
            # REMOVE STALE TRACKING DATA
            # ---------------------------------------------

            for employee_id in list(last_seen):

                if (
                    now_mono - last_seen[employee_id]
                    > STALE_TRACK_SECONDS
                ):

                    confirmed_side.pop(employee_id, None)
                    candidate_side.pop(employee_id, None)
                    candidate_count.pop(employee_id, None)
                    last_seen.pop(employee_id, None)

                    identity_counts[employee_id] = 0

            # ---------------------------------------------
            # AUTOMATIC 6 PM CHECKOUT
            # ---------------------------------------------

            finalize_open_attendance_at_closing(now_dt)

            if now_dt.strftime("%H:%M") >= OFFICE_CLOSE:

                for employee_id in list(employee_state):

                    if employee_state[employee_id] == "IN":
                        employee_state[employee_id] = "LOGGED_OUT"

            # ---------------------------------------------
            # CAMERA INFORMATION
            # ---------------------------------------------

            cv2.putText(
                frame,
                "TAPO C500 | LIVE ATTENDANCE",
                (15, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                now_dt.strftime("%d-%m-%Y %H:%M:%S"),
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            publish_frame(frame)

            if show_window:

                cv2.imshow(
                    "Tapo C500 - CCTV Attendance",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q")):
                    print("[CAMERA] Stop requested.")
                    break

    finally:

        if cap is not None:
            cap.release()

        if show_window:
            cv2.destroyAllWindows()

        with FRAME_LOCK:
            LATEST_JPEG = None

        if stop_event.is_set():
            set_camera_status("STOPPED")
        else:
            # Unexpected normal exit should not appear as LIVE.
            set_camera_status(
                "STOPPED",
                "Camera recognition loop ended."
            )

        print("[CAMERA] CCTV recognition stopped.")


# =========================================================
# 16. FLASK CAMERA WORKER
# =========================================================

def camera_worker():

    try:

        set_camera_status("STARTING")

        print("[CAMERA WORKER] Initializing camera...")

        live_cctv_recognition(
            show_window=False,
            stop_event=CAMERA_STOP_EVENT
        )

    except Exception as error:

        set_camera_status("ERROR", error)

        print("[CAMERA WORKER ERROR]", repr(error))

    finally:

        if CAMERA_STOP_EVENT.is_set():
            set_camera_status("STOPPED")


def start_camera_worker():

    global CAMERA_THREAD

    if (
        CAMERA_THREAD is not None
        and CAMERA_THREAD.is_alive()
    ):

        print("[CAMERA] Worker is already running.")
        return False

    CAMERA_STOP_EVENT.clear()

    set_camera_status("STARTING")

    CAMERA_THREAD = threading.Thread(
        target=camera_worker,
        daemon=True,
        name="TapoCameraWorker"
    )

    CAMERA_THREAD.start()

    print("[CAMERA] Worker thread started.")

    return True


def stop_camera_worker():

    CAMERA_STOP_EVENT.set()

    if (
        CAMERA_THREAD is not None
        and CAMERA_THREAD.is_alive()
    ):

        CAMERA_THREAD.join(timeout=8)

    if (
        CAMERA_THREAD is None
        or not CAMERA_THREAD.is_alive()
    ):

        set_camera_status("STOPPED")
        print("[CAMERA] Worker stopped.")

    else:

        print("[CAMERA] Stop requested; worker is still closing.")

    return get_camera_status()["status"]


# =========================================================
# 17. MAIN MENU
# =========================================================

def main():

    ensure_tables()

    while True:

        print("\n================================")
        print(" CCTV FACE RECOGNITION SYSTEM")
        print("================================")

        print("1. Train Face Recognition Model")
        print("2. Recognize Employee from Image")
        print("3. Live LOGIN / LOGOUT (Tapo C500)")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":

            try:
                train_model()
            except Exception as error:
                print("[TRAINING ERROR]", error)

        elif choice == "2":

            print(
                "Image recognition is not included "
                "in this camera worker version."
            )

        elif choice == "3":

            try:
                live_cctv_recognition(show_window=True)
            except Exception as error:
                print("[ERROR]", error)

        elif choice == "4":

            print("Exiting CCTV Attendance System.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
