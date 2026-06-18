# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import cv2
# from ultralytics import YOLO
# from function.helper import get_thai_character, data_province, split_license_plate_and_province

# # โหลดโมเดล
# vehicle_model = YOLO("model/license_plate.pt")
# plate_model = YOLO("model/data_plate.pt")

# def get_thai_license_plate(image_path):
#     image = cv2.imread(image_path)

#     # ตรวจจับรถยนต์
#     vehicle_results = vehicle_model(image, conf=0.4)
#     detected_classes = []

#     # วาด Bounding Box ของรถ
#     for result in vehicle_results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             car_roi = image[y1:y2, x1:x2]
#             plate_results = plate_model(car_roi, conf=0.3)
#             plates = []
#             for plate in plate_results:
#                 for plate_box in plate.boxes:
#                     px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
#                     px1, px2 = px1 + x1, px2 + x1
#                     py1, py2 = py1 + y1, py2 + y1
#                     plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))

#             plates.sort(key=lambda x: x[0])

#             for plate in plates:
#                 px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
#                 cv2.rectangle(image, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)
#                 clsname = plate_model.names[int(cls)]
#                 detected_classes.append(clsname)
                
#     print(detected_classes)
#     for item in detected_classes:
#         if item in data_province:
#             detected_classes.remove(item)
#             detected_classes.append(item)
#     print(detected_classes)
    
#     combined_text = "".join([get_thai_character(item) for item in detected_classes])
#     print(combined_text)
#     license_plate, province = split_license_plate_and_province(combined_text)
 
#     print(f"ทะเบียนรถ: {license_plate}")
#     print(f"จังหวัด: {province}")

#     return image, license_plate, province

# def browse_image():
#     filepath = filedialog.askopenfilename(title="Select an Image", 
#                                           filetypes=[("JPEG", "*.jpg *.jpeg"),
#                                                     ("PNG", "*.png"),
#                                                     ("All files", "*.*")])
#     if filepath:
#         image, license_plate, province = get_thai_license_plate(filepath)
#         show_image_in_tkinter(image)
#         display_results(license_plate, province)

# def show_image_in_tkinter(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image)

#     # กำหนดขนาดของ Label ที่จะแสดงภาพ
#     max_width = 800
#     max_height = 700

#     # คำนวณอัตราส่วนการปรับขนาด
#     width, height = image.size
#     aspect_ratio = width / height

#     if width > max_width:
#         width = max_width
#         height = int(width / aspect_ratio)
#     if height > max_height:
#         height = max_height
#         width = int(height * aspect_ratio)

#     image = image.resize((width, height), Image.Resampling.LANCZOS)
#     image_tk = ImageTk.PhotoImage(image)

#     panel.config(image=image_tk)
#     panel.image = image_tk

# def display_results(license_plate, province):
#     label_result.config(text=f"ทะเบียนรถ: {license_plate}\nจังหวัด: {province}")

# # สร้างหน้าต่าง Tkinter
# root = tk.Tk()
# root.title("License Plate Detection")
# root.geometry("800x600")
# root.configure(bg="lightgray")

# # เพิ่มปุ่มเพื่อเลือกภาพ
# btn_browse = tk.Button(root, text="เลือกภาพ", command=browse_image, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
# btn_browse.pack(pady=20)

# # เพิ่ม Label แสดงผลลัพธ์
# label_result = tk.Label(root, text="ผลการตรวจจับ", font=("Helvetica", 14), bg="lightgray")
# label_result.pack(pady=10)

# # เพิ่ม Panel สำหรับแสดงภาพ
# panel = tk.Label(root)
# panel.pack(pady=10)

# # เริ่มทำงาน
# root.mainloop()


# gui.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from function.helper import get_thai_character, data_province, split_license_plate_and_province

# 🔥 กัน Tkinter บน macOS งอแง
import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

# โหลดโมเดล
vehicle_model = YOLO("model/license_plate.pt")
plate_model = YOLO("model/data_plate.pt")


def get_thai_license_plate(image_path):
    image = cv2.imread(image_path)

    vehicle_results = vehicle_model(image, conf=0.4)
    detected_classes = []

    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            car_roi = image[y1:y2, x1:x2]
            plate_results = plate_model(car_roi, conf=0.3)

            plates = []
            for plate in plate_results:
                for plate_box in plate.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    px1, px2 = px1 + x1, px2 + x1
                    py1, py2 = py1 + y1, py2 + y1
                    plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))

            plates.sort(key=lambda x: x[0])

            for plate in plates:
                px1, cls, (x1p, y1p, x2p, y2p) = plate
                cv2.rectangle(image, (x1p, y1p), (x2p, y2p), (255, 255, 0), 2)
                clsname = plate_model.names[int(cls)]
                detected_classes.append(clsname)

    # reorder province
    for item in detected_classes[:]:
        if item in data_province:
            detected_classes.remove(item)
            detected_classes.append(item)

    combined_text = "".join([get_thai_character(item) for item in detected_classes])
    license_plate, province = split_license_plate_and_province(combined_text)

    print(f"ทะเบียนรถ: {license_plate}")
    print(f"จังหวัด: {province}")

    return image, license_plate, province


def browse_image():
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("All Files", "*.*")
        ]
    )

    if filepath:
        image, license_plate, province = get_thai_license_plate(filepath)
        show_image_in_tkinter(image)
        display_results(license_plate, province)


def show_image_in_tkinter(image):
    try:
        # 🔥 เก็บ original
        img = image.copy()

        # convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # numpy → PIL
        img = Image.fromarray(img)

        # resize
        img = img.resize((800, 500))
        img = img.copy()

        # PIL → Tkinter
        img_tk = ImageTk.PhotoImage(img)

        panel.config(image=img_tk)
        panel.image = img_tk
        panel.update()

    except Exception as e:
        print("Tkinter แสดงไม่ได้ → fallback OpenCV")
        print(e)

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def display_results(license_plate, province):
    label_result.config(
        text=f"ทะเบียนรถ: {license_plate}\nจังหวัด: {province}"
    )

# ===== UI =====
root = tk.Tk()
root.title("License Plate Detection")
root.geometry("900x700")
root.configure(bg="white")

# 🔥 แสดงทะเบียน (ด้านบนใหญ่ๆ)
label_result = tk.Label(
    root,
    text="ทะเบียนรถ: -\nจังหวัด: -",
    font=("Helvetica", 20, "bold"),
    bg="white",
    fg="black"
)
label_result.pack(pady=20)

# 🔥 ปุ่มเลือกภาพ
btn_browse = tk.Button(
    root,
    text="เลือกภาพ",
    command=browse_image,
    font=("Helvetica", 14),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5
)
btn_browse.pack(pady=10)

# 🔥 panel แสดงภาพ (ด้านล่าง)
panel = tk.Label(root, bg="white")
panel.pack(pady=20)

root.mainloop()