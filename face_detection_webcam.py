import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import winsound

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = YOLO("runs/yoga_pose_detection/weights/best.pt")  # Update path as needed

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# === Define target pose angles and tolerance ===
POSES = {
    "downdog": {"left_hip_angle": (90, 20)},
    "goddess": {"left_knee_angle": (90, 15)},
    "tree": {"right_knee_angle": (60, 15), "left_hip_angle": (170, 10)},
    "warrior2": {"left_knee_angle": (90, 15), "left_elbow_angle": (165, 15)},
    "plank": {"left_elbow_angle": (170, 10), "left_knee_angle": (170, 10)}
}

def angle_accuracy(actual, target, tolerance):
    diff = abs(actual - target)
    return max(0, 100 - (diff / tolerance) * 100)

def calculate_pose_accuracy(angles, target_pose):
    required = POSES[target_pose]
    total_accuracy = 0
    count = 0
    for key, (target_angle, tol) in required.items():
        if key in angles:
            acc = angle_accuracy(angles[key], target_angle, tol)
            total_accuracy += acc
            count += 1
    return total_accuracy / count if count > 0 else 0

def log_pose_result(pose, duration, avg_accuracy):
    with open("pose_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Pose", "Timestamp", "Duration", "Accuracy (%)"])
        writer.writerow([pose, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{duration}s", f"{avg_accuracy:.2f}%"])

# === Select pose ===
print("Available poses:", list(POSES.keys()))
selected_pose = input("Enter pose to perform: ").strip().lower()
if selected_pose not in POSES:
    print("Invalid pose.")
    exit()

# === Webcam ===
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    hold_start = None
    hold_duration = 3
    pose_logged = False
    accuracy_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        results = model.predict(source=frame, conf=0.25, stream=True)
        display_text = f"Pose: {selected_pose}"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                person = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                angles = {}
                accuracy = 0

                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark

                    def get(name):
                        p = lm[mp_pose.PoseLandmark[name].value]
                        return int(p.x * person.shape[1]), int(p.y * person.shape[0])

                    try:
                        shoulder = get("LEFT_SHOULDER")
                        elbow = get("LEFT_ELBOW")
                        wrist = get("LEFT_WRIST")
                        hip = get("LEFT_HIP")
                        knee = get("LEFT_KNEE")
                        ankle = get("LEFT_ANKLE")
                        rknee = get("RIGHT_KNEE")

                        angles = {
                            "left_elbow_angle": calculate_angle(shoulder, elbow, wrist),
                            "left_knee_angle": calculate_angle(hip, knee, ankle),
                            "left_hip_angle": calculate_angle(shoulder, hip, knee),
                            "right_knee_angle": calculate_angle(hip, rknee, ankle)
                        }

                        accuracy = calculate_pose_accuracy(angles, selected_pose)
                        mp_drawing.draw_landmarks(person, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    except Exception as e:
                        print("Error:", e)

                if accuracy >= 70:  # Allow some flexibility in hold accuracy
                    if hold_start is None:
                        hold_start = time.time()
                        accuracy_history = []
                    accuracy_history.append(accuracy)
                    elapsed = time.time() - hold_start
                    if elapsed >= hold_duration and not pose_logged:
                        avg_accuracy = sum(accuracy_history) / len(accuracy_history)
                        display_text = f"✅ Held! Final Accuracy: {avg_accuracy:.1f}%"
                        log_pose_result(selected_pose, hold_duration, avg_accuracy)
                        winsound.Beep(1000, 500)
                        pose_logged = True
                    else:
                        remaining = int(hold_duration - elapsed)
                        display_text = f"Holding... {remaining}s"
                else:
                    hold_start = None
                    accuracy_history = []
                    display_text = f"Accuracy: {accuracy:.1f}%"  # Show live deviation-based accuracy

                color = (0, 255, 0) if accuracy >= 70 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, selected_pose, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, display_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow("Yoga Pose Accuracy", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
