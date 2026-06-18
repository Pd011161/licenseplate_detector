from ultralytics import YOLO

# Load a model
model_path = "C:\\Users\\nara-\\Desktop\\newProject\\7.licenseplate_detector\\yolo11n.pt"
model = YOLO(model_path)
if __name__ == "__main__":
    # Train the model

      # เทรนต่อจากโมเดลเดิม
    train_results = model.train(
        data="C:\\Users\\nara-\\Desktop\\newProject\\7.licenseplate_detector\\datasets\\data.yaml",  # path to dataset YAML
        epochs=200,  # number of training epochs
        imgsz=640,  # training image size
        device="cuda",  # device to run on, i.e. device=0 or device=0,s1,2,3 or device=cpu
 # ทำให้เทรนต่อจากโมเดลเดิม
    )

    # Evaluate model performance on the validation set
    metrics = model.val()
    print("==========================")
    print(metrics)
