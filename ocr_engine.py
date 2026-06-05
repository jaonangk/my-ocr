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

def detect_document_corners(image):
    """
    Smart Document Corner Detection
    ใช้หลายเทคนิค (Multi-strategy) เพื่อหาพิกัด 4 มุมของเอกสาร
    """
    height = image.shape[0]
    width = image.shape[1]
    ratio = height / 800.0
    if ratio < 1: ratio = 1
    dim = (int(width / ratio), int(height / ratio))
    resized = cv2.resize(image, dim)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    img_area = dim[0] * dim[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    v = np.median(cv2.GaussianBlur(gray, (5, 5), 0))
    lower_canny = int(max(0, (1.0 - 0.33) * v))
    upper_canny = int(min(255, (1.0 + 0.33) * v))

    strategies = [
        lambda g: cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), lower_canny, upper_canny),
        lambda g: cv2.Canny(cv2.bilateralFilter(g, 9, 75, 75), 30, 200),
        lambda g: cv2.adaptiveThreshold(cv2.GaussianBlur(g, (5, 5), 0), 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ]

    document_contour = None

    for strat in strategies:
        edged = strat(gray)
        dilated = cv2.dilate(edged, kernel, iterations=2)
        closed = cv2.erode(dilated, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            if cv2.contourArea(c) < img_area * 0.01:
                continue

            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            
            for eps in [0.02, 0.03, 0.04, 0.05, 0.08, 0.1]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    document_contour = approx
                    break
            
            if document_contour is not None:
                break
                
        if document_contour is not None:
            break

    if document_contour is not None:
        pts = document_contour.reshape(4, 2) * ratio
        return order_points(pts)

    # 🚨 แผนสำรอง: Bounding Box
    edged = strategies[0](gray)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    closed = cv2.erode(dilated, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        c = contours[0]
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)
        
        pad_x = int(w * 0.01)
        pad_y = int(h * 0.01)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
        return order_points(pts)

    pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    return order_points(pts)