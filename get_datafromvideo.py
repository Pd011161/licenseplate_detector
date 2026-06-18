import cv2
import re
from ultralytics import YOLO
from function.helper import get_thai_character, data_province, split_license_plate_and_province

# โหลดโมเดล
vehicle_model = YOLO("model/license_plate.pt")  # โมเดลตรวจจับรถ
plate_model = YOLO("model/data_plate.pt")  # โมเดลตรวจจับป้ายทะเบียน
print(plate_model.names)

def get_thai_license_plate_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        # ตรวจจับรถยนต์
        vehicle_results = vehicle_model(frame, conf=0.3,verbose=False)
        detected_classes = []
        
        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียว = รถ
                
                # ครอปบริเวณที่เป็นรถ
                car_roi = frame[y1:y2, x1:x2]
                plate_results = plate_model(car_roi, conf=0.3,verbose=False)
                plates = []
                
                for plate in plate_results:
                    for plate_box in plate.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        px1, px2 = px1 + x1, px2 + x1
                        py1, py2 = py1 + y1, py2 + y1
                        plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))
                
                # เรียงลำดับป้ายทะเบียนตามพิกัด X
                plates.sort(key=lambda x: x[0])
                
                for plate in plates:
                    px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
                    cv2.rectangle(frame, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)  # สีฟ้า = ป้ายทะเบียน
                    clsname = plate_model.names[int(cls)]
                    detected_classes.append(clsname)
        
        # จัดเรียงตัวอักษรบนป้ายทะเบียน
        #print(detected_classes)
        for item in detected_classes:
            if item in data_province:
                detected_classes.remove(item)
                detected_classes.append(item)
        
        combined_text = "".join(get_thai_character(newval) for newval in detected_classes)
        license_plate, province = split_license_plate_and_province(combined_text)
        print("ทะเบียนรถ:", license_plate ,"จังหวัด:", province)
  
        
        # แสดงเฟรมพร้อมการตรวจจับ
        cv2.imshow("License Plate Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ใช้งานฟังก์ชัน
get_thai_license_plate_from_video("video/video1.mp4")
