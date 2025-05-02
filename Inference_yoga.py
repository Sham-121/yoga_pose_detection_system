from ultralytics import YOLO
import os
import glob
import cv2

# === Configuration ===
model_path = r'D:/yoga_pose_detection/runs/yoga_pose_detection/weights/best.pt'  # Trained model
test_image = r'D:/yoga_pose_detection/test_image/downdog.jpg'                 # Make sure this file exists!
test_folder = r'D:/yoga_pose_detection/dataset/test'                             # Folder of test images
use_webcam = False                                                               # Set True for webcam

# === Load Model ===
try:
    model = YOLO(model_path)
    print(f"✅ Loaded trained model from: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# === Inference on a Single Image ===
def predict_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    print(f"🔍 Running inference on image: {image_path}")
    results = model(image_path, show=False, save=True)  # show=False to avoid OpenCV error
    print("📁 Result saved to: runs/detect/predict/")

# === Inference on a Folder of Images ===
def predict_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
    if not image_files:
        print("❌ No images found in folder.")
        return
    print(f"🔍 Found {len(image_files)} images in {folder_path}")
    for image_path in image_files:
        model(image_path, show=False, save=True)
    print("📁 Results saved to: runs/detect/predict/")

# === Real-time Webcam Inference ===
def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Running webcam. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, save=False, show=True, stream=True)
        for r in results:
            pass  # This renders the results

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam closed.")

# === Choose What to Run ===
if use_webcam:
    predict_webcam()
else:
    predict_single_image(test_image)
    # predict_folder(test_folder)  # Uncomment to run batch inference
