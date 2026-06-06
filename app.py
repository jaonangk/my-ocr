import cv2
import io
import re
import tempfile
import os
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RecAipt - Receipt scanning tools",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from llm_engine import call_typhoon_llm
from ocr_engine import (
    deskew_image,
    load_image_or_pdf,
    process_method_4_sharpening,
    run_typhoon_ocr,
    detect_document_corners,
    four_point_transform,
)
from cropper_ui import st_cropper
import numpy as np

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# =========================================================
# GLOBAL NATIVE CSS DESIGN
# =========================================================
st.markdown("""
<style>
header, footer, #MainMenu,
[data-testid="stToolbar"],
[data-testid="stSidebar"] {
    visibility: hidden !important;
    display: none !important;
    height: 0 !important;
}
.stApp { background-color: #FFF2F6 !important; }
.block-container { max-width:100% !important; padding:1.5rem 3rem !important; }

.hero-title {
    text-align:center; color:#4A2E35;
    font-size:32px; font-weight:500; margin:35px 0 10px;
}
.hero-subtitle {
    text-align:center; color:#C29BA4;
    font-size:15px; margin-bottom:45px;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    max-width:780px !important; margin:0 auto !important; display:block !important;
}
[data-testid="stFileUploaderDropzone"] {
    background:#FFFFFF !important;
    border:2px dashed #F4C6D5 !important;
    border-radius:28px !important;
    min-height:220px !important;
    display:flex !important; flex-direction:column !important;
    align-items:center !important; justify-content:center !important;
    padding:40px 30px !important;
    box-shadow:0 12px 35px rgba(74,46,53,0.03) !important;
    cursor:pointer !important;
}
[data-testid="stFileUploaderDropzone"] svg { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] {
    display:flex !important; flex-direction:column !important;
    align-items:center;
}
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content:"📄"; font-size:44px; line-height:1;
    margin-bottom:14px; display:block;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content:"Choose or paste a file here (image or PDF)";
    color:#A3858C; font-size:15px; display:block;
    margin-top:10px; text-align:center;
}
[data-testid="stFileUploader"] label { display:none !important; }
[data-testid="stFileUploaderDropzoneInputButton"] {
    opacity:0 !important; position:absolute !important;
    width:100% !important; height:100% !important;
    top:0 !important; left:0 !important; cursor:pointer !important;
}

/* ── Result wrapper ── */
.result-wrapper {
    background:#FFFFFF; border-radius:32px; padding:35px;
    max-width:1450px; margin:0 auto !important;
    box-shadow:0 12px 40px rgba(74,46,53,0.04);
}
[data-testid="stHorizontalBlock"] { gap:30px !important; align-items:flex-start !important; }
.img-card-wrap {
    background:#F5F5F5; border-radius:24px;
    overflow:hidden; border:1px solid #F8D7E3;
}
[data-testid="stHtml"] { padding:0 !important; margin:0 !important; }
iframe { display:block !important; margin:0 auto !important; border-radius:24px !important; }
[data-testid="stElementToolbar"] {display:none !important;}
button[title="View fullscreen"] {display:none !important;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS & DATE FORMAT NORMALIZER
# =========================================================
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def safe_float(value, default=0.0):
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default

def safe_int(value, default=1):
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return default

def normalize_date(date_str):
    if not date_str: return None
    date_str = str(date_str).strip()
    thai_months = {
        "มกราคม": "01", "กุมภาพันธ์": "02", "มีนาคม": "03", "เมษายน": "04",
        "พฤษภาคม": "05", "มิถุนายน": "06", "กรกฎาคม": "07", "สิงหาคม": "08",
        "กันยายน": "09", "ตุลาคม": "10", "พฤศจิกายน": "11", "ธันวาคม": "12"
    }
    try:
        match_numeric = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', date_str)
        if match_numeric:
            day, month, year = int(match_numeric.group(1)), int(match_numeric.group(2)), int(match_numeric.group(3))
            if year > 2500: year -= 543
            return f"{year:04d}-{month:02d}-{day:02d}"
        for m_name, m_num in thai_months.items():
            if m_name in date_str:
                nums = re.findall(r'\d+', date_str)
                if len(nums) >= 2:
                    day, year = int(nums[0]), int(nums[-1])
                    if year > 2500: year -= 543
                    return f"{year:04d}-{m_num}-{day:02d}"
        return date_str
    except Exception:
        return date_str

# =========================================================
# SVG ICONS
# =========================================================
SVG_BOX = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C97D98" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>'

# =========================================================
# DETAIL CARD (iframe-based for HTML layout)
# =========================================================
def build_detail_card_html(extracted_json):
    merchant = extracted_json.get("seller", {}).get("name", extracted_json.get("store_name", "—")) or "—"
    receipt_no = extracted_json.get("document_number", extracted_json.get("receipt_no", "—")) or "—"
    date_val = extracted_json.get("document_date", extracted_json.get("date", "—")) or "—"
    receipt_type = extracted_json.get("document_type", "ใบเสร็จรับเงิน/ใบกำกับภาษี") or "ใบเสร็จรับเงิน/ใบกำกับภาษี"
    items_list = extracted_json.get("items", []) or []
    subtotal_val = safe_float(extracted_json.get("amount_before_tax", extracted_json.get("subtotal", 0)))
    vat_val = safe_float(extracted_json.get("vat_amount", extracted_json.get("vat", 0)))
    total_val = safe_float(extracted_json.get("grand_total", extracted_json.get("total", 0)))

    rows_html = ""
    for idx, item in enumerate(items_list):
        name = item.get("item_description", item.get("name", "—"))
        qty = safe_int(item.get("quantity", item.get("qty", 1)))
        price = safe_float(item.get("unit_price", 0))
        amt = safe_float(item.get("subtotal", qty * price))
        rows_html += f"<tr><td class='num'>{idx + 1}</td><td>{name}</td><td>{qty}</td><td>{price:,.2f}</td><td style='text-align:right'>{amt:,.2f}</td></tr>"

    if not rows_html:
        rows_html = '<tr><td colspan="5" style="text-align:center;color:#C29BA4;padding:16px 0">ไม่พบรายการสินค้า</td></tr>'

    subtotal_row = f'<div class="t-row"><span>ยอดก่อน VAT :</span><span>{subtotal_val:,.2f} บาท</span></div>' if subtotal_val else ""
    vat_row = f'<div class="t-row"><span>VAT :</span><span>{vat_val:,.2f} บาท</span></div>' if vat_val else ""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent;padding:2px}}
svg{{display:inline-block;vertical-align:middle;flex-shrink:0}}
.card{{background:#FFF6F8;border-radius:24px;border:1px solid #F8D7E3;padding:24px 28px}}
.dc-title{{font-size:17px;font-weight:700;color:#4A2E35;text-align:center;margin-bottom:20px}}
.badge{{display:inline-block;background:#FFF0F5;color:#A35271;border:1px solid #F4C6D5;border-radius:8px;font-size:12px;padding:4px 12px;margin-bottom:16px}}
.info-row{{display:flex;gap:8px;font-size:13px;margin-bottom:12px;align-items:baseline}}
.lbl{{color:#7A5A63;min-width:130px;flex-shrink:0;font-weight:500}}
.val{{color:#4A2E35;font-weight:600}}
.divider{{border:none;border-top:1px solid #F4E0E8;margin:16px 0}}
.sec-lbl{{font-size:13px;color:#4A2E35;font-weight:bold;margin-bottom:12px;display:flex;align-items:center;gap:6px}}
.tbl{{width:100%;border-collapse:collapse;font-size:13px}}
.tbl th{{color:#C29BA4;font-weight:400;padding:4px 6px 9px;border-bottom:1px solid #F4E0E8;text-align:center}}
.tbl th:nth-child(2){{text-align:left}}
.tbl td{{padding:8px 6px;color:#4A2E35;text-align:center;border-bottom:1px dashed #FFF0F5}}
.tbl td:nth-child(2){{text-align:left}}
.num{{color:#C29BA4;font-size:12px}}
.totals{{padding-top:14px}}
.t-row{{display:flex;justify-content:space-between;font-size:13px;color:#A07A85;margin-bottom:8px}}
.grand{{color:#4A2E35;font-weight:700;font-size:15px}}
</style>
</head><body>
<div class="card">
  <div class="dc-title">รายละเอียดเอกสาร (OCR Categorized)</div>
  <div class="dc-body">
    <span class="badge">{receipt_type}</span>
    <div class="info-row"><span class="lbl">หัวข้อ / ร้านค้า :</span><span class="val">{merchant}</span></div>
    <div class="info-row"><span class="lbl">เลขที่เอกสาร :</span><span class="val">{receipt_no}</span></div>
    <div class="info-row"><span class="lbl">วันที่ :</span><span class="val">{date_val}</span></div>
    <hr class="divider">
    <div class="sec-lbl">{SVG_BOX} รายละเอียดที่สกัดได้</div>
    <table class="tbl">
      <thead><tr>
        <th style="width:32px">ลำดับ</th><th>รายการ</th>
        <th style="width:50px">จำนวน</th><th style="width:60px">ราคา</th>
        <th style="width:60px;text-align:right">รวม</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <hr class="divider">
    <div class="totals">
      {subtotal_row}{vat_row}
      <div class="t-row grand"><span>ยอดรวมทั้งหมด :</span><span>{total_val:,.2f} บาท</span></div>
    </div>
  </div>
</div>
</body></html>"""

# =========================================================
# PDF EXPORT
# =========================================================
def generate_pdf(processed_img, extracted_json):
    """Build a PDF with cropped image on page 1 and extracted text on page 2."""
    if not HAS_FPDF:
        raise ImportError("fpdf2 not installed")

    merchant = extracted_json.get("seller", {}).get("name", extracted_json.get("store_name", "—")) or "—"
    receipt_no = extracted_json.get("document_number", extracted_json.get("receipt_no", "—")) or "—"
    date_val = extracted_json.get("document_date", extracted_json.get("date", "—")) or "—"
    receipt_type = extracted_json.get("document_type", "Receipt/Tax Invoice") or "Receipt/Tax Invoice"
    items_list = extracted_json.get("items", []) or []
    subtotal_val = safe_float(extracted_json.get("amount_before_tax", extracted_json.get("subtotal", 0)))
    vat_val = safe_float(extracted_json.get("vat_amount", extracted_json.get("vat", 0)))
    total_val = safe_float(extracted_json.get("grand_total", extracted_json.get("total", 0)))

    # Convert image to RGB JPEG bytes
    display_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB) if len(
        processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    _, img_buf = cv2.imencode('.jpg', display_img)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(img_buf.tobytes())
        tmp_path = tmp.name

    try:
        pdf = FPDF()

        # --- Page 1: Image ---
        pdf.add_page()
        img_h, img_w = processed_img.shape[:2]
        page_w = pdf.w - 20
        page_h_avail = pdf.h - 20
        scale = min(page_w / img_w, page_h_avail / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        x_offset = (pdf.w - draw_w) / 2
        pdf.image(tmp_path, x=x_offset, y=10, w=draw_w, h=draw_h)

        # --- Page 2: Text ---
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'Document OCR Result', ln=True, align='C')
        pdf.ln(4)

        def safe_str(s):
            return str(s).encode('latin-1', 'replace').decode('latin-1')

        rows = [
            ('Type', receipt_type),
            ('Merchant', merchant),
            ('Document No', receipt_no),
            ('Date', date_val),
        ]
        for label, val in rows:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(50, 8, f'{label}:', ln=False)
            pdf.set_font('Helvetica', '', 11)
            pdf.cell(0, 8, safe_str(val), ln=True)

        pdf.ln(4)
        pdf.set_draw_color(200, 180, 190)
        pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
        pdf.ln(6)

        # Items table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(255, 240, 245)
        pdf.cell(12, 8, '#', border=1, fill=True, align='C')
        pdf.cell(80, 8, 'Description', border=1, fill=True)
        pdf.cell(18, 8, 'Qty', border=1, fill=True, align='C')
        pdf.cell(30, 8, 'Unit Price', border=1, fill=True, align='R')
        pdf.cell(0, 8, 'Amount', border=1, fill=True, align='R', ln=True)

        pdf.set_font('Helvetica', '', 10)
        for idx, item in enumerate(items_list):
            name = item.get("item_description", item.get("name", "—"))
            qty = safe_int(item.get("quantity", item.get("qty", 1)))
            price = safe_float(item.get("unit_price", 0))
            amt = safe_float(item.get("subtotal", qty * price))
            pdf.cell(12, 7, str(idx + 1), border='B', align='C')
            pdf.cell(80, 7, safe_str(name), border='B')
            pdf.cell(18, 7, str(qty), border='B', align='C')
            pdf.cell(30, 7, f'{price:,.2f}', border='B', align='R')
            pdf.cell(0, 7, f'{amt:,.2f}', border='B', align='R', ln=True)

        pdf.ln(4)
        if subtotal_val:
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, f'Subtotal (before VAT): {subtotal_val:,.2f}', align='R', ln=True)
        if vat_val:
            pdf.cell(0, 7, f'VAT: {vat_val:,.2f}', align='R', ln=True)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 9, f'Total: {total_val:,.2f}', align='R', ln=True)

        return pdf.output(dest='S').encode('latin-1')
    finally:
        os.unlink(tmp_path)

# =========================================================
# PAGE ROUTING (STATE MACHINE)
# =========================================================
if "app_phase" not in st.session_state:
    st.session_state["app_phase"] = "UPLOAD"

# ---------------------------------------------------------
# PHASE 1 : UPLOAD
# ---------------------------------------------------------
if st.session_state["app_phase"] == "UPLOAD":
    st.markdown("<div class='hero-title'>OCR Document Categorizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>ถ่ายภาพเอกสารมุมกว้าง ระบบจะพยายามหาขอบให้ และให้คุณปรับแก้ด้วยตัวเองได้ก่อนสแกน</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"], key="uploader_widget", label_visibility="collapsed")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        with st.spinner("⏳ กำลังวิเคราะห์และหาขอบเอกสาร..."):
            img = load_image_or_pdf(file_bytes, file_name)
            if img is None:
                st.error("❌ Unsupported file")
                st.stop()

            pts = detect_document_corners(img)
            st.session_state["orig_img"] = img
            st.session_state["detected_pts"] = pts
            st.session_state["app_phase"] = "CROP"
            st.rerun()

# ---------------------------------------------------------
# PHASE 2 : INTERACTIVE CROP
# ---------------------------------------------------------
elif st.session_state["app_phase"] == "CROP":
    orig_img = st.session_state["orig_img"]
    pts = st.session_state["detected_pts"]

    # st_cropper returns None until the user confirms
    final_pts = st_cropper(orig_img, pts, key="cropper")

    if final_pts is not None:
        final_pts = np.array(final_pts, dtype="float32")

        with st.spinner("⏳ กำลังครอบตัด แก้เอียง และเพิ่มความชัด..."):
            warped = four_point_transform(orig_img, final_pts)
            deskewed = deskew_image(warped)
            processed = process_method_4_sharpening(deskewed)
            st.session_state["processed_img"] = processed

        with st.spinner("⚡ กำลังอ่านข้อความด้วย OCR..."):
            raw_text = run_typhoon_ocr(st.session_state["processed_img"])
            st.session_state["raw_text"] = raw_text

        if "[ERROR]" in raw_text or not raw_text.strip():
            st.error("❌ OCR failed")
            if st.button("ลองใหม่"):
                reset_app()
                st.rerun()
        else:
            with st.spinner("🤖 กำลังจัดหมวดหมู่ข้อมูล (Categorizing)..."):
                extracted_json = call_typhoon_llm(raw_text)
                st.session_state["extracted_json"] = extracted_json
                st.session_state["app_phase"] = "RESULT"
            st.rerun()

# ---------------------------------------------------------
# PHASE 3 : RESULT DASHBOARD
# ---------------------------------------------------------
elif st.session_state["app_phase"] == "RESULT":
    processed_img = st.session_state["processed_img"]
    raw_text = st.session_state["raw_text"]
    extracted_json = st.session_state["extracted_json"]

    has_error = (isinstance(extracted_json, dict) and "error" in extracted_json and extracted_json["error"])

    st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="img-card-wrap">', unsafe_allow_html=True)
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB) if len(
            processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Back button — native Streamlit (no iframe, no blank white box)
        if st.button("← กลับ", key="back_btn", type="secondary"):
            reset_app()
            st.rerun()

    with col_right:
        if has_error:
            st.error(f"❌ {extracted_json['error']}")
        else:
            items_count = len(extracted_json.get("items", []) or [])
            # Card height: base 300px + 40px per item row + fixed totals section
            card_height = 300 + (items_count * 40) + 100
            components.html(build_detail_card_html(extracted_json), height=card_height, scrolling=False)

            # Export PDF — native download button (no iframe)
            if HAS_FPDF:
                try:
                    pdf_bytes = generate_pdf(processed_img, extracted_json)
                    st.download_button(
                        label="⬇️ ส่งออก PDF (ภาพ + รายละเอียด)",
                        data=pdf_bytes,
                        file_name="ocr_document.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="export_pdf_btn"
                    )
                except Exception as e:
                    st.warning(f"ไม่สามารถสร้าง PDF ได้: {e}")
            else:
                st.info("ติดตั้ง fpdf2 เพื่อใช้ฟีเจอร์ส่งออก PDF: `pip install fpdf2`")

    st.markdown('</div>', unsafe_allow_html=True)