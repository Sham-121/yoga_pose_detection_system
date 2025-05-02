import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import os
import winsound  # Works on Windows. For Linux/Mac, use `os.system("play beep.wav")` or similar.

# === Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = YOLO("runs/yoga_pose_detection/weights/best.pt")  # Update to your model path

# === Calculate Angle ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# === Pose Rules ===
POSES = {
    "downdog": {"left_hip_angle": (70, 110)},
    "goddess": {"left_knee_angle": (80, 100)},
    "tree": {"right_knee_angle": (40, 70), "left_hip_angle": (160, 180)},
    "warrior2": {"left_knee_angle": (80, 100), "left_elbow_angle": (150, 180)},
    "plank": {"left_elbow_angle": (160, 180), "left_knee_angle": (160, 180)}
}

# === Pose Detection ===
def detect_pose(angles, target_pose):
    rules = POSES.get(target_pose, {})
    match_count = 0
    total = len(rules)
    for joint, (min_angle, max_angle) in rules.items():
        if joint in angles and min_angle <= angles[joint] <= max_angle:
            match_count += 1
    accuracy = (match_count / total) * 100 if total else 0
    return accuracy

# === CSV Logger ===
def log_pose_result(pose, duration, accuracy):
    file_path = "pose_results.csv"
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Pose", "Timestamp", "Duration", "Accuracy (%)"])
        writer.writerow([pose, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{duration} seconds", f"{accuracy:.2f}%"])

# === User Selection ===
print("Available Poses:", list(POSES.keys()))
selected_pose = input("Enter the pose you want to perform: ").strip().lower()

if selected_pose not in POSES:
    print("Invalid pose selected.")
    exit()

print(f"Now get ready to perform: {selected_pose}")

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set full resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    hold_start_time = None
    hold_duration = 10  # seconds
    pose_held_successfully = False
    final_accuracy = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_text = "Detecting..."
        results = model.predict(source=frame, conf=0.25, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.shape[0] < 100 or person_crop.shape[1] < 100:
                    continue

                rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                angles = {}
                current_accuracy = 0

                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark

                    def get_point(name):
                        pt = lm[mp_pose.PoseLandmark[name].value]
                        return int(pt.x * person_crop.shape[1]), int(pt.y * person_crop.shape[0])

                    try:
                        shoulder = get_point("LEFT_SHOULDER")
                        elbow = get_point("LEFT_ELBOW")
                        wrist = get_point("LEFT_WRIST")
                        hip = get_point("LEFT_HIP")
                        knee = get_point("LEFT_KNEE")
                        ankle = get_point("LEFT_ANKLE")
                        rknee = get_point("RIGHT_KNEE")

                        angles = {
                            "left_elbow_angle": calculate_angle(shoulder, elbow, wrist),
                            "left_knee_angle": calculate_angle(hip, knee, ankle),
                            "left_hip_angle": calculate_angle(shoulder, hip, knee),
                            "right_knee_angle": calculate_angle(hip, rknee, ankle)
                        }

                        current_accuracy = detect_pose(angles, selected_pose)

                        mp_drawing.draw_landmarks(person_crop, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    except Exception as e:
                        print("Pose detection error:", e)

                # Timer & Accuracy logic
                if current_accuracy >= 90:
                    if hold_start_time is None:
                        hold_start_time = time.time()
                    else:
                        elapsed = time.time() - hold_start_time
                        remaining = hold_duration - int(elapsed)
                        if elapsed >= hold_duration:
                            final_accuracy = current_accuracy
                            pose_held_successfully = True
                        else:
                            display_text = f"Holding '{selected_pose}'... {remaining}s left ({current_accuracy:.1f}%)"
                else:
                    hold_start_time = None
                    display_text = f"❌ Re-hold '{selected_pose}' (Accuracy: {current_accuracy:.1f}%)"

                # Draw bounding box and label
                color = (0, 255, 0) if current_accuracy >= 90 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{selected_pose}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if pose_held_successfully:
            display_text = f"✅ Held '{selected_pose}' for {hold_duration}s with {final_accuracy:.1f}% accuracy"
            winsound.Beep(1000, 500)  # Beep after success
            log_pose_result(selected_pose, hold_duration, final_accuracy)
            cv2.putText(frame, display_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Yoga Pose Timer", frame)
            cv2.waitKey(2000)
            break

        # Draw status text
        cv2.putText(frame, display_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Yoga Pose Timer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
