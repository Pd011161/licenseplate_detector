# Version 1.0.0
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import re
from ultralytics import YOLO
from function.helper import get_thai_character, split_license_plate_and_province
import base64

app = FastAPI()

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vehicle_model = YOLO("model/license_plate.pt")
plate_model   = YOLO("model/data_plate.pt")


# ═══════════════════════════════════════════════════════════════════════════════
#  THAI PLATE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# 77 จังหวัดของไทย
VALID_PROVINCES = {
    "กระบี่", "กรุงเทพมหานคร", "กาญจนบุรี", "กาฬสินธุ์",
    "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี",
    "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่",
    "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม",
    "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี",
    "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี",
    "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา",
    "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี",
    "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร",
    "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง",
    "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน",
    "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
    "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
    "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
    "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี",
    "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"
}

_C = r"[\u0E01-\u0E4E]"   # Unicode range พยัญชนะไทย

# ┌─────────────────────────────────────────────────────────────────┐
# │  รูปแบบทะเบียนที่รองรับ                                          │
# │  (1) มาตรฐาน   : พยัญชนะ 2 ตัว  + เลข 1-4 หลัก  → กข 1234    │
# │  (2) หมวดใหม่  : เลข 1 หลัก + พยัญชนะ 2 ตัว + เลข 1-4 หลัก   │
# │                  → 1กข 1234                                     │
# │  (3) ป้ายประมูล: พยัญชนะ 3 ตัว + เลข 1-4 หลัก  → กขค 1234    │
# │                  (ต้องมีจังหวัด มิฉะนั้น reject)                  │
# └─────────────────────────────────────────────────────────────────┘
_RE_STANDARD  = re.compile(rf"^{_C}{{2}}\d{{1,4}}$")          # กข1234
_RE_NEW_SERIES = re.compile(rf"^\d{_C}{{2}}\d{{1,4}}$")        # 1กข1234
_RE_AUCTION   = re.compile(rf"^{_C}{{3}}\d{{1,4}}$")           # กขค1234 (ประมูล)
_RE_ANY_DIGITS = re.compile(r"\d+")


def is_valid_thai_plate(license_plate: str, province: str) -> tuple[bool, str]:
    """
    ตรวจสอบความถูกต้องของทะเบียนไทย
    คืนค่า (is_valid, reason)

    กฎหลัก
    -------
    1. ต้องมีทั้งทะเบียนและจังหวัด
    2. จังหวัดต้องอยู่ในรายการ 77 จังหวัด
    3. ตัวเลขต้องไม่เกิน 4 หลัก (≥ 5 หลัก = ป้ายปลอมแน่นอน)
    4. ตัวเลขต้องไม่เป็น 0 ทั้งหมด
    5. รูปแบบต้องตรงใดตรงหนึ่ง:
       • มาตรฐาน   [พยัญชนะ 2 ตัว][เลข 1-4]         กข1234
       • หมวดใหม่  [เลข 1 หลัก][พยัญชนะ 2 ตัว][เลข 1-4]  1กข1234
       • ป้ายประมูล [พยัญชนะ 3 ตัว][เลข 1-4]  (ต้องมีจังหวัด) กขค1234
    6. พยัญชนะ 3 ตัวโดยไม่มีจังหวัด = ผิด Logic ทันที
    """
    # ── 1. ต้องมีครบ ───────────────────────────────────────────────
    if not license_plate:
        return False, "ไม่พบทะเบียน"
    if not province:
        return False, "ไม่พบจังหวัด"

    plate = license_plate.strip()
    prov  = province.strip()

    # ── 2. จังหวัดถูกต้อง ──────────────────────────────────────────
    if prov not in VALID_PROVINCES:
        return False, f"จังหวัดไม่ถูกต้อง: '{prov}'"

    # ── 3. ตัวเลขต้องไม่เกิน 4 หลัก ───────────────────────────────
    for m in _RE_ANY_DIGITS.finditer(plate):
        if len(m.group()) >= 5:
            return False, f"ตัวเลขเกิน 4 หลัก ({m.group()}) → ป้ายปลอม"

    # ── 4. ตัวเลขต้องไม่เป็น 0 ทั้งหมด ────────────────────────────
    digits_only = "".join(re.findall(r"\d", plate))
    if digits_only and int(digits_only) == 0:
        return False, "เลขทะเบียนเป็น 0 ทั้งหมด"

    # ── 5 & 6. ตรวจ pattern ────────────────────────────────────────
    if _RE_STANDARD.match(plate):
        return True, "มาตรฐาน (พยัญชนะ 2 + เลข 1-4)"

    if _RE_NEW_SERIES.match(plate):
        return True, "หมวดใหม่ (เลข 1 + พยัญชนะ 2 + เลข 1-4)"

    if _RE_AUCTION.match(plate):
        # พยัญชนะ 3 ตัว = ป้ายประมูลพิเศษ → ยอมรับได้เฉพาะมีจังหวัด (ตรวจไปแล้วในข้อ 2)
        return True, "ป้ายประมูล (พยัญชนะ 3 + เลข 1-4)"

    return False, f"รูปแบบไม่ตรงกฎทะเบียนไทย: '{plate}'"


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLAP / DUPLICATE BOX DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _box_overlap_ratio(a: tuple, b: tuple) -> float:
    """
    Overlap ratio = intersection / min(area_a, area_b)
    ใช้ min แทน union เพราะกรอบอักษรขนาดต่างกัน
    กรอบเล็กซ้อนอยู่ในกรอบใหญ่ก็ถือว่าทับกัน
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def has_overlapping_boxes(boxes: list, threshold: float = 0.4) -> tuple:
    """
    ตรวจว่ามีกรอบคู่ไหนทับกันเกิน threshold หรือไม่
    boxes = list of (x1, y1, x2, y2)
    คืนค่า (has_overlap, reason)
    """
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ratio = _box_overlap_ratio(boxes[i], boxes[j])
            print(f"Overlap ratio between box {i+1} and {j+1}: {ratio:.2f}")
            if ratio >= threshold:
                return True, (
                    f"กรอบที่ {i+1} และ {j+1} ทับกัน "
                    f"(overlap {ratio:.0%}) → detect ซ้ำ"
                )
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(image):
    """
    Returns:
        image     – annotated image (bounding boxes drawn)
        vehicles  – list of validated { license_plate, province, bbox }
        rejected  – list of rejected  { raw_text, reason, bbox }
    """
    vehicles = []
    rejected = []

    vehicle_results = vehicle_model(image, conf=0.5, imgsz=1280) #0.4
    print(f"Detected {len(vehicle_results)} vehicles in the image")
    for box in vehicle_results[0].boxes:
        print(f"Vehicle conf: {box.conf[0]:.2f}")

    for result in vehicle_results:
        for box in result.boxes:
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
            car_roi = image[vy1:vy2, vx1:vx2]

            plate_results = plate_model(car_roi, conf=0.6) #0.3
            print(f"Detected {len(plate_results)} plate characters in this vehicle box")
            for box in plate_results[0].boxes:
                print(f"Plate conf: {box.conf[0]:.2f}")
            # รวม character ของรถคันนี้
            plates = []
            for plate in plate_results:
                for plate_box in plate.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    px1 += vx1; px2 += vx1
                    py1 += vy1; py2 += vy1
                    plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))

            # เรียงซ้าย→ขวา (reading order)
            plates.sort(key=lambda x: x[0])

            vehicle_classes = []
            char_bboxes     = []
            for _, cls, (x1p, y1p, x2p, y2p) in plates:
                cv2.rectangle(image, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                vehicle_classes.append(plate_model.names[int(cls)])
                char_bboxes.append((x1p, y1p, x2p, y2p))

            if not vehicle_classes:
                continue

            # ── 1. ตรวจกรอบทับกัน (ก่อนอื่นเลย) ──────────────────────────────
            overlap, overlap_reason = has_overlapping_boxes(char_bboxes)
            if overlap:
                cv2.rectangle(image, (vx1, vy1), (vx2, vy2), (50, 50, 220), 2)
                rejected.append({
                    "raw_text": " ".join(vehicle_classes),
                    "reason":   overlap_reason,
                    "bbox":     [vx1, vy1, vx2, vy2],
                })
                continue

            n_boxes = len(vehicle_classes)

            combined_text = "".join([get_thai_character(c) for c in vehicle_classes])
            license_plate, province = split_license_plate_and_province(combined_text)

            print(f"Combined Text: {combined_text}")
            print(f"License Plate: {license_plate}, Province: {province}")

            # ── 2. Cross-check จำนวนกรอบ vs ตัวอักษรในผลลัพธ์ ─────────────────
            n_chars = len(license_plate)
            if province:
                n_boxes=n_boxes-1
            print(f"Number of Characters: {n_chars}, Number of Boxes: {n_boxes}")
            if n_chars != (n_boxes):
                cv2.rectangle(image, (vx1, vy1), (vx2, vy2), (50, 50, 220), 2)
                rejected.append({
                    "raw_text": combined_text,
                    "reason":   (
                        f"จำนวนกรอบ ({n_boxes}) "
                        f"≠ ตัวอักษรในผลลัพธ์ ({n_chars}) "
                        f"→ detect ไม่ครบหรือซ้ำ"
                    ),
                    "bbox": [vx1, vy1, vx2, vy2],
                })
                continue

            # ── 3. Validate รูปแบบทะเบียนไทย ──────────────────────────────────
            valid, reason = is_valid_thai_plate(license_plate, province)

            if valid:
                cv2.rectangle(image, (vx1, vy1), (vx2, vy2), (0, 200, 80), 2)
                # cv2.putText(
                #     image, license_plate,
                #     (vx1, max(vy1 - 8, 20)),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2, cv2.LINE_AA,
                # )
                vehicles.append({
                    "license_plate": license_plate,
                    "province":      province,
                    "bbox":          [vx1, vy1, vx2, vy2],
                })
            else:
                cv2.rectangle(image, (vx1, vy1), (vx2, vy2), (50, 50, 220), 2)
                rejected.append({
                    "raw_text": combined_text,
                    "reason":   reason,
                    "bbox":     [vx1, vy1, vx2, vy2],
                })

    print("Rejected:", rejected)
    return image, vehicles, rejected


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    image, vehicles, rejected = process_image(image)   # รับ 3 ค่า

    _, buffer  = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode()

    return {
        "vehicles": vehicles,   # ทะเบียนที่สมบูรณ์และถูกต้อง
        "count":    len(vehicles),
        "rejected": rejected,   # detect ได้แต่ไม่ผ่าน validation (debug)
        "image":    img_base64,
    }


import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8180,
        reload=True,
        log_level="info",
    )


# Version 2.0.0
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import re
# from ultralytics import YOLO
# from function.helper import get_thai_character, split_license_plate_and_province
# import base64

# app = FastAPI()

# # ─────────────────────────────────────────────────────────────
# # CORS
# # ─────────────────────────────────────────────────────────────
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------------------------------------
# # MODEL
# # license_plate.pt = detect กรอบทะเบียน
# # data_plate.pt    = detect ตัวอักษร
# # -------------------------------------------------------------
# vehicle_model = YOLO("model/license_plate.pt")
# plate_model   = YOLO("model/data_plate.pt")


# # ═════════════════════════════════════════════════════════════
# # THAI PLATE VALIDATION
# # ═════════════════════════════════════════════════════════════
# VALID_PROVINCES = {
#     "กระบี่", "กรุงเทพมหานคร", "กาญจนบุรี", "กาฬสินธุ์",
#     "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี",
#     "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่",
#     "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม",
#     "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี",
#     "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี",
#     "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา",
#     "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี",
#     "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร",
#     "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง",
#     "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน",
#     "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
#     "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
#     "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
#     "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี",
#     "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"
# }

# _C = r"[\u0E01-\u0E4E]"

# _RE_STANDARD   = re.compile(rf"^{_C}{{2}}\d{{1,4}}$")
# _RE_NEW_SERIES = re.compile(rf"^\d{_C}{{2}}\d{{1,4}}$")
# _RE_AUCTION    = re.compile(rf"^{_C}{{3}}\d{{1,4}}$")
# _RE_ANY_DIGITS = re.compile(r"\d+")


# def is_valid_thai_plate(license_plate: str, province: str):

#     if not license_plate:
#         return False, "ไม่พบทะเบียน"

#     if not province:
#         return False, "ไม่พบจังหวัด"

#     plate = license_plate.strip()
#     prov  = province.strip()

#     if prov not in VALID_PROVINCES:
#         return False, f"จังหวัดไม่ถูกต้อง"

#     for m in _RE_ANY_DIGITS.finditer(plate):
#         if len(m.group()) >= 5:
#             return False, "เลขเกิน 4 หลัก"

#     digits_only = "".join(re.findall(r"\d", plate))
#     if digits_only and int(digits_only) == 0:
#         return False, "เลขเป็น 0"

#     if _RE_STANDARD.match(plate):
#         return True, "standard"

#     if _RE_NEW_SERIES.match(plate):
#         return True, "new"

#     if _RE_AUCTION.match(plate):
#         return True, "auction"

#     return False, "format invalid"


# # ═════════════════════════════════════════════════════════════
# # OVERLAP CHECK
# # ═════════════════════════════════════════════════════════════
# def _box_overlap_ratio(a, b):

#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b

#     ix1 = max(ax1, bx1)
#     iy1 = max(ay1, by1)
#     ix2 = min(ax2, bx2)
#     iy2 = min(ay2, by2)

#     inter_w = max(0, ix2 - ix1)
#     inter_h = max(0, iy2 - iy1)

#     inter = inter_w * inter_h

#     if inter == 0:
#         return 0

#     area_a = (ax2 - ax1) * (ay2 - ay1)
#     area_b = (bx2 - bx1) * (by2 - by1)

#     return inter / min(area_a, area_b)


# def has_overlapping_boxes(boxes, threshold=0.4):

#     for i in range(len(boxes)):
#         for j in range(i + 1, len(boxes)):

#             ratio = _box_overlap_ratio(boxes[i], boxes[j])

#             if ratio >= threshold:
#                 return True, "detect ซ้ำ"

#     return False, ""


# # ═════════════════════════════════════════════════════════════
# # IMAGE PROCESSING
# # ═════════════════════════════════════════════════════════════
# def process_image(image):

#     vehicles = []
#     rejected = []

#     # ---------------------------------------------
#     # รถเร็ว detect ไม่ค่อยติด
#     # แก้โดย imgsz สูงขึ้น + conf ลดลง
#     # ---------------------------------------------
#     vehicle_results = vehicle_model(
#         image,
#         conf=0.5,
#         imgsz=1280
#     )

#     print(f"Detected {len(vehicle_results)} vehicles")

#     for result in vehicle_results:
#         for box in result.boxes:

#             vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])

#             # -------------------------------------------------
#             # crop เฉพาะทะเบียน
#             # -------------------------------------------------
#             car_roi = image[vy1:vy2, vx1:vx2]

#             if car_roi.size == 0:
#                 continue
            
#             # -------------------------------------------------
#             # detect ตัวอักษรต่อ
#             # -------------------------------------------------
#             plate_results = plate_model(
#                 car_roi,
#                 conf=0.6
#             )

#             plates = []

#             for plate in plate_results:
#                 for plate_box in plate.boxes:

#                     px1, py1, px2, py2 = map(int, plate_box.xyxy[0])

#                     px1 += vx1
#                     px2 += vx1
#                     py1 += vy1
#                     py2 += vy1

#                     plates.append(
#                         (
#                             px1,
#                             plate_box.cls,
#                             (px1, py1, px2, py2)
#                         )
#                     )

#             # ซ้าย -> ขวา
#             plates.sort(key=lambda x: x[0])

#             vehicle_classes = []
#             char_bboxes = []

#             for _, cls, (x1p, y1p, x2p, y2p) in plates:

#                 cv2.rectangle(
#                     image,
#                     (x1p, y1p),
#                     (x2p, y2p),
#                     (0, 255, 0),
#                     2
#                 )

#                 vehicle_classes.append(
#                     plate_model.names[int(cls)]
#                 )

#                 char_bboxes.append(
#                     (x1p, y1p, x2p, y2p)
#                 )

#             if not vehicle_classes:
#                 continue

#             # overlap
#             overlap, overlap_reason = has_overlapping_boxes(char_bboxes)

#             if overlap:
#                 rejected.append({
#                     "raw_text": "".join(vehicle_classes),
#                     "reason": overlap_reason,
#                     "bbox": [vx1, vy1, vx2, vy2]
#                 })
#                 continue

#             combined_text = "".join(
#                 [get_thai_character(c) for c in vehicle_classes]
#             )

#             license_plate, province = split_license_plate_and_province(
#                 combined_text
#             )

#             n_boxes = len(vehicle_classes)
#             n_chars = len(license_plate)

#             if province:
#                 n_boxes -= 1

#             if n_chars != n_boxes:
#                 rejected.append({
#                     "raw_text": combined_text,
#                     "reason": "detect ไม่ครบ",
#                     "bbox": [vx1, vy1, vx2, vy2]
#                 })
#                 continue

#             valid, reason = is_valid_thai_plate(
#                 license_plate,
#                 province
#             )

#             if valid:

#                 cv2.rectangle(
#                     image,
#                     (vx1, vy1),
#                     (vx2, vy2),
#                     (0, 200, 80),
#                     2
#                 )

#                 vehicles.append({
#                     "license_plate": license_plate,
#                     "province": province,
#                     "bbox": [vx1, vy1, vx2, vy2]
#                 })

#             else:

#                 cv2.rectangle(
#                     image,
#                     (vx1, vy1),
#                     (vx2, vy2),
#                     (50, 50, 220),
#                     2
#                 )

#                 rejected.append({
#                     "raw_text": combined_text,
#                     "reason": reason,
#                     "bbox": [vx1, vy1, vx2, vy2]
#                 })

#     return image, vehicles, rejected


# # ═════════════════════════════════════════════════════════════
# # ENDPOINT
# # ═════════════════════════════════════════════════════════════
# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):

#     contents = await file.read()

#     np_arr = np.frombuffer(contents, np.uint8)

#     image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     image, vehicles, rejected = process_image(image)

#     _, buffer = cv2.imencode(".jpg", image)

#     img_base64 = base64.b64encode(buffer).decode()

#     return {
#         "vehicles": vehicles,
#         "count": len(vehicles),
#         "rejected": rejected,
#         "image": img_base64
#     }


# # ═════════════════════════════════════════════════════════════
# # RUN
# # ═════════════════════════════════════════════════════════════
# if __name__ == "__main__":

#     import uvicorn

#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8180,
#         reload=True,
#         log_level="info",
#     )