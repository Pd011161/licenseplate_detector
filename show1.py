from ultralytics import YOLO

# Load a model
model = YOLO("model/best_licens.pt")

# Perform object detection on an image with confidence threshold
results = model("car.jpg", conf=0.1)

# Show the results
results[0].show()
