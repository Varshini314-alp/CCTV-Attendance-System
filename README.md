# AI-Based CCTV Employee Attendance Management System

## Project Overview

The AI-Based CCTV Employee Attendance Management System is a Python-based application that automates employee attendance using CCTV cameras and face recognition technology.

The system identifies registered employees, records their entry and exit times, stores attendance information in an SQLite database, and provides a web-based dashboard using Flask.

## Key Features

- **Face Recognition:** Identifies registered employees using OpenCV and LBPH.
- **CCTV Integration:** Connects to TP-Link Tapo C500 CCTV cameras through RTSP.
- **Automatic Attendance:** Records employee check-in and check-out events.
- **Employee Management:** Supports employee registration and face training.
- **Database Management:** Stores employee and attendance records using SQLite.
- **Live Dashboard:** Displays CCTV video and attendance information through Flask.
- **Attendance Reports:** Generates daily attendance reports in XML and CSV formats.
- **Break Tracking:** Maintains employee break records.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| OpenCV | Image processing |
| LBPH | Face recognition |
| Haar Cascade | Face detection |
| YOLO | Object detection |
| Flask | Web application |
| SQLite | Database |
| HTML, CSS | Frontend |
| XML, CSV | Report generation |
| RTSP | CCTV video streaming |

## Project Structure

```text
CCTV_Attendance_System/
│
├── app.py
├── db.py
├── detection.py
├── face_recognition.py
├── reports.py
├── requirements.txt
│
├── employees/
│   └── training_photos/
│
├── database/
│   └── attendance.db
│
├── models/
│   ├── face_model.yml
│   └── labels.json
│
├── templates/
│   ├── register.html
│   ├── employees.html
│   ├── attendance.html
│   └── reports.html
│
├── reports/
│
├── haarcascade_frontalface_default.xml
├── yolo11n.pt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Varshini314-alp/CCTV-Attendance-System.git
```

### 2. Navigate to the project folder

```bash
cd CCTV-Attendance-System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure CCTV

Set your Tapo camera RTSP URL using an environment variable.

```powershell
$env:TAPO_RTSP_URL = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"
```

Replace the placeholders with your camera details. Never publish your camera credentials.

### 7. Run the application

```bash
python app.py
```

Open the dashboard in your browser:

```text
http://127.0.0.1:5000
```

## Attendance Workflow

1. Register employees with their details and photographs.
2. Train the face recognition model.
3. Connect the CCTV camera.
4. Detect and recognize employees.
5. Record employee entry and exit events.
6. Store attendance information in SQLite.
7. Generate attendance reports.

## Future Enhancements

- Multiple CCTV camera integration.
- Centralized monitoring of multiple office locations.
- Improved face recognition accuracy.
- Advanced employee analytics.
- Cloud-based attendance management.
- Role-based authentication and access control.

## Author

**Varshini M**

AI/ML Project Developer

GitHub: https://github.com/Varshini314-alp

## License

This project is developed for educational and demonstration purposes.
