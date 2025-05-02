import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time

# === Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = YOLO("runs/yoga_pose_detection/weights/best.pt")  # Your trained YOLOv10 detection model

# === Angle Calculation ===
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

# === Pose Matching ===
def detect_pose(angles):
    for pose, rules in POSES.items():
        if all(j in angles and rules[j][0] <= angles[j] <= rules[j][1] for j in rules):
            return pose
    return "Wrong Pose"

# === Start Webcam ===
cap = cv2.VideoCapture(0)
prev_time = time.time()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        results = model.predict(source=frame, conf=0.25, stream=True)

        for r in results:
            print("YOLO boxes detected:", len(r.boxes))
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"Box: ({x1}, {y1}) to ({x2}, {y2})")

                # Extract crop
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.shape[0] < 100 or person_crop.shape[1] < 100:
                    print("Skipped small crop.")
                    continue

                # MediaPipe
                rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                angles = {}
                status = "Unknown"

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

                        # Compute Angles
                        angles = {
                            "left_elbow_angle": calculate_angle(shoulder, elbow, wrist),
                            "left_knee_angle": calculate_angle(hip, knee, ankle),
                            "left_hip_angle": calculate_angle(shoulder, hip, knee),
                            "right_knee_angle": calculate_angle(hip, rknee, ankle)
                        }
                        print("Angles:", angles)

                        # Detect Pose
                        status = detect_pose(angles)
                        print("Detected Pose:", status)

                        # Draw keypoints
                        mp_drawing.draw_landmarks(person_crop, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # Display Angles
                        y_offset = 20
                        for k, v in angles.items():
                            cv2.putText(person_crop, f"{k}: {int(v)}", (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            y_offset += 15

                    except Exception as e:
                        print("Angle calculation error:", e)

                # Draw Bounding Box
                color = (0, 255, 0) if status != "Wrong Pose" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Paste back crop with drawings
                resized_crop = cv2.resize(person_crop, (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = resized_crop

        # Show FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show Output
        cv2.imshow("Yoga Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
