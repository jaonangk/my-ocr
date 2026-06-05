import cv2
import fitz  # PyMuPDF
import numpy as np
import requests
import streamlit as st

def load_image_or_pdf(file_bytes, file_name):
    """Load a receipt image, or render the first page of a PDF as an image."""
    if file_name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def deskew_image(image):
    """Estimate and correct small receipt rotation angles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 10 or abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

def process_method_4_sharpening(img):
    """Apply grayscale denoising and sharpening, matching Method 4."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(denoised, -1, kernel)

def run_typhoon_ocr(image_np):
    """Compress and send the preprocessed image to Typhoon OCR API."""
    url = "https://api.opentyphoon.ai/v1/ocr"
    
    # 🔑 ดึง API Key อย่างปลอดภัยจาก Secrets ของ Streamlit
    api_key = st.secrets["OPENTYPHOON_API_KEY"]

    _, encoded_img = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 75])
    image_bytes = encoded_img.tobytes()

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "typhoon-ocr",
        "task_type": "default",
        "temperature": 0,
        "max_tokens": 4096,
        "prompt": (
            "กรุณาดึงข้อความทั้งหมดที่ปรากฏในภาพออกมาให้ครบถ้วนและแม่นยำที่สุด "
            "ห้ามข้ามหรือตัดข้อความใดๆ ทิ้งเด็ดขาด พิมพ์ออกมาตามที่เห็นในภาพ"
        ),
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=data,
            files={"file": ("receipt.jpg", image_bytes, "image/jpeg")},
            timeout=45,
        )
        response.raise_for_status()
        result = response.json()

        texts = []
        for page in result.get("results", []):
            if page.get("success"):
                texts.append(page["message"]["choices"][0]["message"]["content"])
        return "\n".join(texts)
    except Exception as exc:
        return f"[ERROR] ตัวเอนจินคลาวด์ OCR ปฏิเสธการส่งกลับเนื่องจาก: {str(exc)}"

def order_points(pts):
    """เรียงพิกัด 4 จุด: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """ปรับมุมมองภาพ (Perspective Warp) ให้ตั้งตรงและตัดพื้นหลังออก"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def _score_quad(approx, img_area):
    """
    ให้คะแนน quadrilateral โดยพิจารณา:
    - ขนาดพื้นที่เทียบกับรูปทั้งหมด (ควรใหญ่แต่ไม่เต็ม)
    - ความสมมาตร / aspect ratio ที่สมเหตุสมผล
    - จำนวนจุดเท่ากับ 4 พอดี
    """
    if len(approx) != 4:
        return 0.0
    pts = approx.reshape(4, 2).astype("float32")
    quad_area = cv2.contourArea(approx)
    area_ratio = quad_area / img_area
    # ต้องครอบคลุมอย่างน้อย 8% และไม่เกิน 97% ของภาพ
    if area_ratio < 0.08 or area_ratio > 0.97:
        return 0.0
    # ตรวจ aspect ratio ของ bounding rect (ใบเสร็จมักสูงกว่ากว้าง 1:1.5 – 1:6)
    x, y, w, h = cv2.boundingRect(approx)
    ar = max(w, h) / max(min(w, h), 1)
    if ar > 8:
        return 0.0
    return area_ratio  # ยิ่งใหญ่ = ยิ่งดี (ในขอบเขตที่เหมาะสม)


def _find_best_quad(contours, img_area):
    """
    วนลองหาสี่เหลี่ยม 4 มุมที่ดีที่สุดจาก contour list
    ลอง epsilon หลายค่าเพื่อรับมือกับขอบกระดาษยับ / แสงไม่สม่ำเสมอ
    """
    best_approx = None
    best_score = 0.0
    epsilons = [0.02, 0.03, 0.04, 0.05, 0.06]

    # เรียง contour จากใหญ่ → เล็ก และเอาแค่ top-5
    sorted_c = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in sorted_c:
        if cv2.contourArea(c) < img_area * 0.06:
            continue
        peri = cv2.arcLength(c, True)
        for eps in epsilons:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            score = _score_quad(approx, img_area)
            if score > best_score:
                best_score = score
                best_approx = approx

    return best_approx, best_score


def _build_edge_map(gray):
    """สร้าง edge map หลายวิธีแล้ว OR รวมกัน เพื่อรับมือกับแสงทุกสภาพ"""
    h, w = gray.shape

    # --- วิธี 1: CLAHE + Canny ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred1 = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # ใช้ median-based threshold อัตโนมัติ (แทนค่าตายตัว 30/150)
    med = float(np.median(blurred1))
    lo = max(0, int(0.4 * med))
    hi = min(255, int(1.2 * med))
    canny1 = cv2.Canny(blurred1, lo, hi)

    # --- วิธี 2: Canny ค่า sigma ต่างกัน (จับขอบละเอียด) ---
    blurred2 = cv2.GaussianBlur(gray, (9, 9), 0)
    canny2 = cv2.Canny(blurred2, 20, 80)

    # --- วิธี 3: Adaptive Threshold (เอาขอบตัวอักษร+กระดาษ) ---
    adapt = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (7, 7), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # รวม edge map ทั้ง 3 วิธี
    combined = cv2.bitwise_or(canny1, cv2.bitwise_or(canny2, adapt))

    # Morphological close เชื่อมเส้นขาด
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def crop_document(image):
    """
    ตัดขอบเอกสารออกจากพื้นหลัง — Multi-Strategy Pipeline
    1. สร้าง edge map รวม (CLAHE + Canny + Adaptive)
    2. ค้นหา contour ที่ใหญ่ที่สุดและลองหา quad 4 มุม (หลาย epsilon)
    3. ถ้าได้ quad → Perspective Transform (ตรงสุด)
    4. ถ้าไม่ได้ quad → Bounding Box crop พร้อม padding
    5. ถ้าครอบพื้นที่ > 90% ของรูป (ภาพถ่ายชิดมาก) → คืนต้นฉบับ
    """
    orig = image.copy()
    img_area = image.shape[0] * image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- ขั้นที่ 1: สร้าง edge map แบบรวม ----
    edge_map = _build_edge_map(gray)

    # ---- ขั้นที่ 2: หา contours ----
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return orig

    # ---- ขั้นที่ 3: หา quad 4 มุม ----
    best_approx, best_score = _find_best_quad(contours, img_area)

    if best_approx is not None and best_score > 0:
        pts = best_approx.reshape(4, 2).astype("float32")
        quad_area = cv2.contourArea(best_approx)
        # ถ้า quad ครอบ > 90% → ภาพถ่ายชิดมาก ไม่จำเป็นต้อง warp
        if quad_area / img_area > 0.90:
            return orig
        warped = four_point_transform(orig, pts)
        return warped

    # ---- ขั้นที่ 4: Fallback — Bounding Box ของ contour ใหญ่สุด ----
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < img_area * 0.05:
        return orig  # ไม่มีเอกสารชัดเจน

    x, y, w, h = cv2.boundingRect(c)
    # ถ้า bounding box ครอบ > 90% → คืนต้นฉบับ
    if (w * h) / img_area > 0.90:
        return orig

    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    return orig[y1:y2, x1:x2]