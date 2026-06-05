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

def crop_document(image):
    """ค้นหาขอบใบเสร็จแบบโหด (อัปเกรดแผนสำรอง ตัดโต๊ะทิ้ง 100%)"""
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. จับขอบแบบ Canny (จับเส้นขอบที่ตัดกันชัดเจน เช่น ขอบกระดาษขาวกับโต๊ะดำ)
    edged = cv2.Canny(blurred, 30, 150)
    
    # 2. ขยายเส้นขอบให้เชื่อมติดกันเป็นก้อนเดียว (ป้องกันเส้นขาดจากแสงเงา)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    
    # 3. หาก้อน (Contours) ทั้งหมดที่เป็นเส้นรอบนอก
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return orig
        
    # 4. เลือกก้อนที่มีพื้นที่ "ใหญ่ที่สุด" ในรูป (ต้องเป็นเอกสารแน่นอน)
    c = max(contours, key=cv2.contourArea)
    
    # 5. กรองขยะ (ถ้าก้อนใหญ่สุดยังเล็กกว่า 5% ของภาพ แสดงว่าไม่ใช่เอกสาร)
    img_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(c) < (img_area * 0.05):
        return orig

    # 6. พยายามลากเส้นแบบ 4 มุมก่อน (Perspective Transform)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True) # เพิ่มความยืดหยุ่นให้มุม
    
    if len(approx) == 4:
        warped = four_point_transform(orig, approx.reshape(4, 2))
        return warped
    else:
        # 🚨 แผนสำรอง! (Fallback): ถ้าขอบกระดาษยับและไม่เป็น 4 มุมเป๊ะ
        # ให้ครอบกล่องสี่เหลี่ยม (Bounding Box) ล้อมรอบใบเสร็จนั้นแล้วตัดโต๊ะทิ้งเลย!
        x, y, w, h = cv2.boundingRect(c)
        pad = 15 # เผื่อขอบกระดาษไว้ 15 px กันตัวหนังสือขาด
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        return orig[y1:y2, x1:x2]