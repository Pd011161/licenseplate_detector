import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from function.helper import get_thai_character, data_province, split_license_plate_and_province

# โหลดโมเดล
vehicle_model = YOLO("model/license_plate.pt")
plate_model = YOLO("model/data_plate.pt")

image_files = []  # เก็บรายการไฟล์ภาพ

def get_thai_license_plate(image_path):
    image = cv2.imread(image_path)

    # ตรวจจับรถยนต์
    vehicle_results = vehicle_model(image, conf=0.4)
    detected_classes = []

    # วาด Bounding Box ของรถ
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
                px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
                cv2.rectangle(image, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)
                clsname = plate_model.names[int(cls)]
                detected_classes.append(clsname)
    
    
    print(detected_classes)
    for item in detected_classes:
        print(item)
        if item in data_province:
            detected_classes.remove(item)
            detected_classes.append(item)
    print(detected_classes)            
    
    combined_text = "".join([get_thai_character(item) for item in detected_classes])
    license_plate, province = split_license_plate_and_province(combined_text)

    return image, license_plate, province

def browse_images():
    filepaths = filedialog.askopenfilenames(title="Select Images", filetypes=[("JPEG", "*.jpg *.jpeg"),
                                                    ("PNG", "*.png"),
                                                    ("All files", "*.*")])
    if filepaths:
        image_files.extend(filepaths)  # เพิ่มไฟล์ลงใน list
        listbox.delete(0, tk.END)  # ล้างข้อมูลเก่า
        for idx, file in enumerate(image_files, start=1):
            listbox.insert(tk.END, f"{idx}. {file.split('/')[-1]}")  # เพิ่มตัวเลขลำดับ

def on_image_select(event):
    selected_index = listbox.curselection()
    if selected_index:
        index = selected_index[0]
        filepath = image_files[index]
        image, license_plate, province = get_thai_license_plate(filepath)
        show_image_in_tkinter(image)
        display_results(index + 1, license_plate, province)  # ส่งลำดับภาพไปแสดงผล

def show_image_in_tkinter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    max_width = 700
    max_height = 700

    width, height = image.size
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)
    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image)

    panel.config(image=image_tk)
    panel.image = image_tk

def display_results(index, license_plate, province):
    label_result.config(text=f"รายการที่เลือก: {index}\nทะเบียนรถ: {license_plate}\nจังหวัด: {province}")

# สร้างหน้าต่าง Tkinter
root = tk.Tk()
root.title("License Plate Detection")
root.geometry("1000x800")
root.configure(bg="#f0f0f0")

# สร้าง Frame สำหรับ Listbox (ซ้าย)
frame_left = tk.Frame(root, bg="#e0e0e0", width=300, height=600)
frame_left.pack(side="left", fill="y")

# ปุ่มเลือกไฟล์
btn_browse = tk.Button(frame_left, text="เลือกภาพ", command=browse_images, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
btn_browse.pack(pady=20)

# Listbox แสดงรายการภาพ
listbox = tk.Listbox(frame_left, font=("Helvetica", 12), width=30, height=20)
listbox.pack(pady=10)
listbox.bind("<<ListboxSelect>>", on_image_select)

# สร้าง Frame สำหรับแสดงภาพและผลลัพธ์ (ขวา)
frame_right = tk.Frame(root, bg="white", width=700, height=600)
frame_right.pack(side="right", fill="both", expand=True)

# Panel แสดงภาพ
panel = tk.Label(frame_right, bg="white")
panel.pack(pady=20)

# Label แสดงผลลัพธ์
label_result = tk.Label(frame_right, text="ผลการตรวจจับ", font=("Helvetica", 14), bg="white", fg="black")
label_result.pack(pady=10)

# เริ่มทำงาน
root.mainloop()
