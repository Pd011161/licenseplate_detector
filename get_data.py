import cv2
import re
from ultralytics import YOLO
from function.helper import get_thai_character,data_province,split_license_plate_and_province
# โหลดโมเดล

 
# โหลดโมเดล
vehicle_model = YOLO("model/license_plate.pt")

# โมเดลตรวจจับป้ายทะเบียน
plate_model = YOLO("model/data_plate.pt")
print(plate_model.names)


def get_thai_license_plate(image):
    # โหลดภาพ
    image = cv2.imread(image)

    # ตรวจจับรถยนต์
    vehicle_results = vehicle_model(image,conf=0.3)
    
    detected_classes = []
    # วาด Bounding Box ของรถ
    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียว = รถ

            # ครอปเฉพาะบริเวณที่เป็นรถ
            car_roi = image[y1:y2, x1:x2]
            # ตรวจจับป้ายทะเบียน
            plate_results = plate_model(car_roi,conf=0.3)
           
            plates = []
            # เก็บผลการตรวจจับป้ายทะเบียน
            for plate in plate_results:
                for plate_box in plate.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    # แปลงพิกัดเป็นของภาพหลัก
                    px1, px2 = px1 + x1, px2 + x1
                    py1, py2 = py1 + y1, py2 + y1

                    # เก็บข้อมูลตำแหน่งและข้อมูลของป้ายทะเบียน
                    plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))  # เก็บ X, class, Bounding Box
     

            # จัดเรียงป้ายทะเบียนตามพิกัด X (จากซ้ายไปขวา)
            plates.sort(key=lambda x: x[0])
     
            # วาด Bounding Box ของป้ายทะเบียนตามลำดับ
            for plate in plates:
                px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
                cv2.rectangle(image, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)  # สีฟ้า = ป้ายทะเบียน
               
                clsname = plate_model.names[int(cls)]  # หรือ plate_box.class_name ขึ้นอยู่กับเวอร์ชันที่ใช้งาน
                detected_classes.append(clsname)  # แปลงรหัสเป็นตัวอักษรไทย
    
    print("---------------------------------------------------")
    print(detected_classes)
    print("---------------------------------------------------")

    
    for item in detected_classes:
        if item in data_province: 
            detected_classes.remove(item)  # ลบจังหวัดออก
            detected_classes.append(item)   # เพิ่มจังหวัดไว้ท้ายสุด

    print(f" ที่ถูกต้อง {detected_classes}")
    
    
    combined_text = ""
    for newval in detected_classes:
        thai_character = get_thai_character(newval)
        combined_text += thai_character
        
        
    print(combined_text)
    license_plate, province = split_license_plate_and_province(combined_text)
    print("ทะเบียนรถ:", license_plate)
    print("จังหวัด:", province)

    # แสดงภาพ
    cv2.imshow("Detection and OCR", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


get_thai_license_plate("myphoto/photo5.jpg")
