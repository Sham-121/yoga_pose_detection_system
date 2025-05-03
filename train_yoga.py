from ultralytics import YOLO
import os
import json

# Configuration
data_yaml = r'D:/yoga_pose_detection/data.yaml'        # Path to data.yaml
weights = r'D:/yoga_pose_detection/yolov10n.pt'        # Pretrained YOLOv10 nano weights
output_dir = r'D:/yoga_pose_detection/runs'            # Output directory for results
epochs = 100                                           # Training for 100 epochs
batch_size = 8                                         # Batch size
img_size = 640                                         # Image resolution
device = 'cuda:0'                                      # GPU ('cuda:0') or CPU ('cpu')

def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv10 model
    try:
        model = YOLO(weights)
        print(f"✅ Loaded model from {weights}")
    except FileNotFoundError:
        print(f"❌ Weights file not found at {weights}. Please download yolov10n.pt.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Train the model
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=output_dir,
            name='yoga_pose_detection',
            exist_ok=True,
            verbose=True
        )
        print(f"✅ Training completed. Results saved to {output_dir}/yoga_pose_detection")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    # Validate the model
    try:
        metrics = model.val(
            data=data_yaml,
            imgsz=img_size,
            batch=batch_size,
            device=device
        )
        print("✅ Validation completed.")
        print("📊 Validation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        # Save validation results to a JSON file
        metrics_path = os.path.join(output_dir, 'yoga_pose_detection', 'val_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"📁 Metrics saved to {metrics_path}")

    except Exception as e:
        print(f"❌ Error during validation: {e}")

if __name__ == '__main__':
    main()