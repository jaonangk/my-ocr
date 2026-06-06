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
    page_title="RecAipt — OCR Document Categorizer",
    page_icon="📄",
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

# =========================================================
# GLOBAL CSS — Fully Responsive (Mobile + Tablet + Desktop)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

header, footer, #MainMenu,
[data-testid="stToolbar"],
[data-testid="stSidebar"] {
    visibility: hidden !important;
    display: none !important;
    height: 0 !important;
}
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: linear-gradient(145deg,#FEF0F5 0%,#FFF5F8 40%,#FEF7FF 100%) !important;
    font-family: 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
}
.block-container { max-width:100% !important; padding:0 2.5rem 2rem !important; }

/* ── Header ── */
.app-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:16px 28px;
    background:rgba(255,255,255,0.88);
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    border-bottom:1px solid rgba(201,125,152,0.15);
    position:sticky; top:0; z-index:100;
    flex-wrap:wrap; gap:8px;
}
.app-logo { display:flex; align-items:center; gap:10px; }
.app-logo-icon {
    width:36px; height:36px; flex-shrink:0;
    background:linear-gradient(135deg,#E8829F,#C97D98);
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:18px; box-shadow:0 4px 12px rgba(201,125,152,0.4);
}
.app-logo-text {
    font-size:20px; font-weight:800; letter-spacing:-0.5px; line-height:1.2;
    background:linear-gradient(135deg,#A35271,#C97D98);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.app-logo-sub { font-size:11px; color:#C29BA4; font-weight:400; }
.header-badge {
    background:linear-gradient(135deg,#FFF0F5,#FFE4EE);
    border:1px solid #F4C6D5; color:#A35271;
    font-size:12px; font-weight:600;
    padding:6px 14px; border-radius:20px; white-space:nowrap;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    max-width:720px !important; margin:0 auto !important; display:block !important;
}
[data-testid="stFileUploaderDropzone"] {
    background:rgba(255,255,255,0.95) !important;
    border:2px dashed #E8A8C0 !important;
    border-radius:28px !important;
    min-height:200px !important;
    display:flex !important; flex-direction:column !important;
    align-items:center !important; justify-content:center !important;
    padding:40px 20px !important;
    box-shadow:0 16px 48px rgba(201,125,152,0.08) !important;
    cursor:pointer !important;
    transition:border-color .2s,box-shadow .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color:#C97D98 !important;
    box-shadow:0 20px 56px rgba(201,125,152,0.14) !important;
}
[data-testid="stFileUploaderDropzone"] svg { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]>div>span,
[data-testid="stFileUploaderDropzoneInstructions"]>div>small { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] {
    display:flex !important; flex-direction:column !important; align-items:center;
}
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content:"📄"; font-size:44px; line-height:1;
    margin-bottom:14px; display:block;
    filter:drop-shadow(0 4px 8px rgba(201,125,152,0.3));
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content:"วางหรือคลิกเพื่ออัปโหลดภาพเอกสาร หรือ PDF";
    color:#B5909A; font-size:14px; display:block;
    margin-top:10px; text-align:center;
    font-family:'Inter',sans-serif; font-weight:400;
}
[data-testid="stFileUploader"] label { display:none !important; }
[data-testid="stFileUploaderDropzoneInputButton"] {
    opacity:0 !important; position:absolute !important;
    width:100% !important; height:100% !important;
    top:0 !important; left:0 !important; cursor:pointer !important;
}

/* ── Result card wrapper (uses st.columns = stHorizontalBlock) ── */
[data-testid="stHorizontalBlock"] {
    background:rgba(255,255,255,0.92);
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    border-radius:28px; padding:28px;
    max-width:1450px; margin:0 auto !important;
    box-shadow:0 20px 60px rgba(74,46,53,0.07),0 4px 16px rgba(74,46,53,0.04);
    gap:24px !important; align-items:flex-start !important;
    border:1px solid rgba(255,255,255,0.8);
}
.img-card-wrap {
    background:linear-gradient(145deg,#F8F4F5,#F2ECEE);
    border-radius:18px; overflow:hidden;
    border:1px solid rgba(201,125,152,0.2);
    box-shadow:inset 0 2px 8px rgba(74,46,53,0.04);
}

/* ── Back button ── */
[data-testid="stBaseButton-secondary"] {
    background:rgba(255,255,255,0.9) !important;
    border:1.5px solid #E8A8C0 !important;
    color:#A35271 !important; border-radius:12px !important;
    font-weight:600 !important; font-size:13px !important;
    padding:8px 20px !important;
    box-shadow:0 2px 8px rgba(201,125,152,0.12) !important;
    transition:all .2s !important;
    margin-top:10px !important; width:100% !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background:#FFF0F5 !important; border-color:#C97D98 !important;
    box-shadow:0 4px 14px rgba(201,125,152,0.22) !important;
    transform:translateY(-1px) !important;
}

[data-testid="stSpinner"] { color:#C97D98 !important; }
[data-testid="stHtml"] { padding:0 !important; margin:0 !important; }
iframe { display:block !important; width:100% !important; border:none !important; }
[data-testid="stElementToolbar"] { display:none !important; }
button[title="View fullscreen"] { display:none !important; }

/* ═══════════════════════════════════════════
   RESPONSIVE — Tablet  (≤ 900px)
═══════════════════════════════════════════ */
@media screen and (max-width:900px) {
    .block-container { padding:0 1rem 1.5rem !important; }
    .app-header { padding:12px 16px; }
    .app-logo-text { font-size:17px; }
    .app-logo-sub { display:none; }
    .header-badge { font-size:11px; padding:5px 10px; }

    /* Stack result columns vertically */
    [data-testid="stHorizontalBlock"] {
        flex-direction:column !important;
        border-radius:20px !important;
        padding:18px !important;
        gap:16px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width:100% !important; flex:none !important; min-width:100% !important;
    }
    [data-testid="stFileUploader"] { max-width:100% !important; }
    [data-testid="stFileUploaderDropzone"] {
        min-height:160px !important;
        border-radius:22px !important;
        padding:28px 16px !important;
    }
}

/* ═══════════════════════════════════════════
   RESPONSIVE — Mobile  (≤ 600px)
═══════════════════════════════════════════ */
@media screen and (max-width:600px) {
    .block-container { padding:0 0.4rem 1rem !important; }
    .app-header { padding:10px 12px; }
    .app-logo-icon { width:30px; height:30px; font-size:15px; }
    .app-logo-text { font-size:15px; }
    .header-badge { font-size:10px; padding:4px 8px; border-radius:14px; }

    [data-testid="stHorizontalBlock"] {
        border-radius:16px !important;
        padding:12px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        min-height:140px !important;
        border-radius:18px !important;
        padding:22px 12px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::before { font-size:36px; }
    [data-testid="stFileUploaderDropzoneInstructions"]::after { font-size:12px; }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
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
# DETAIL CARD HTML (iframe — responsive inside too)
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
        rows_html += (
            f"<tr>"
            f"<td class='num'>{idx+1}</td>"
            f"<td class='name'>{name}</td>"
            f"<td>{qty}</td>"
            f"<td>{price:,.2f}</td>"
            f"<td class='amt'>{amt:,.2f}</td>"
            f"</tr>"
        )

    if not rows_html:
        rows_html = '<tr><td colspan="5" class="empty">ไม่พบรายการสินค้า</td></tr>'

    subtotal_row = f'<div class="t-row"><span>ยอดก่อน VAT :</span><span>{subtotal_val:,.2f} บาท</span></div>' if subtotal_val else ""
    vat_row = f'<div class="t-row"><span>VAT :</span><span>{vat_val:,.2f} บาท</span></div>' if vat_val else ""

    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:auto;overflow:hidden}}
body{{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:transparent;padding:2px;
  font-size:14px;
}}
svg{{display:inline-block;vertical-align:middle;flex-shrink:0}}
.card{{
  background:linear-gradient(160deg,#FFF8FA,#FFF4F7);
  border-radius:20px;border:1px solid rgba(201,125,152,0.2);
  padding:20px 22px 18px;
  box-shadow:0 4px 20px rgba(74,46,53,0.06);
}}
.dc-title{{font-size:15px;font-weight:700;color:#3D1F28;text-align:center;margin-bottom:16px}}
.badge{{
  display:inline-block;
  background:linear-gradient(135deg,#FFF0F5,#FFE8F0);
  color:#A35271;border:1px solid rgba(201,125,152,0.35);
  border-radius:20px;font-size:11px;font-weight:600;
  padding:4px 12px;margin-bottom:14px;
}}
.info-row{{display:flex;gap:6px;font-size:13px;margin-bottom:9px;align-items:flex-start;line-height:1.5;flex-wrap:wrap}}
.lbl{{color:#9A7080;min-width:120px;flex-shrink:0;font-weight:500;font-size:12.5px}}
.val{{color:#3D1F28;font-weight:600;flex:1;min-width:0;word-break:break-word}}
.divider{{border:none;border-top:1px solid rgba(201,125,152,0.18);margin:12px 0}}
.sec-lbl{{font-size:11.5px;color:#5C3040;font-weight:700;margin-bottom:10px;display:flex;align-items:center;gap:5px;text-transform:uppercase;letter-spacing:0.4px}}
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{color:#B899A5;font-weight:500;padding:4px 4px 7px;border-bottom:1px solid rgba(201,125,152,0.2);text-align:center;font-size:11px}}
.tbl th.name-h{{text-align:left}}
.tbl td{{padding:7px 4px;color:#3D1F28;text-align:center;border-bottom:1px solid rgba(255,240,245,0.8);vertical-align:top}}
.name{{text-align:left!important;font-weight:500;word-break:break-word}}
.tbl tr:last-child td{{border-bottom:none}}
.num{{color:#C9A0B0;font-size:11px;font-weight:400}}
.amt{{text-align:right!important;font-weight:600;color:#5C3040}}
.empty{{text-align:center;color:#C29BA4;padding:14px 0;font-size:12px}}
.totals{{
  background:linear-gradient(135deg,rgba(255,240,245,0.6),rgba(255,248,250,0.6));
  border-radius:12px;padding:12px 14px;margin-top:4px;
}}
.t-row{{display:flex;justify-content:space-between;font-size:12.5px;color:#9A7080;margin-bottom:6px}}
.t-row:last-child{{margin-bottom:0}}
.grand{{color:#3D1F28;font-weight:700;font-size:14px;border-top:1.5px solid rgba(201,125,152,0.25);padding-top:9px;margin-top:4px}}
/* Mobile inside iframe */
@media (max-width:400px){{
  .card{{padding:14px 14px 12px;border-radius:16px}}
  .lbl{{min-width:90px;font-size:11.5px}}
  .val{{font-size:12.5px}}
  .tbl{{font-size:11px}}
  .tbl th{{font-size:10px}}
}}
</style>
</head><body>
<div class="card" id="card">
  <div class="dc-title">📋 รายละเอียดเอกสาร (OCR)</div>
  <span class="badge">{receipt_type}</span>
  <div class="info-row"><span class="lbl">หัวข้อ / ร้านค้า :</span><span class="val">{merchant}</span></div>
  <div class="info-row"><span class="lbl">เลขที่เอกสาร :</span><span class="val">{receipt_no}</span></div>
  <div class="info-row"><span class="lbl">วันที่ :</span><span class="val">{date_val}</span></div>
  <hr class="divider">
  <div class="sec-lbl">{SVG_BOX}&nbsp;รายละเอียดที่สกัดได้</div>
  <table class="tbl">
    <thead><tr>
      <th style="width:26px">No.</th>
      <th class="name-h">รายการ</th>
      <th style="width:40px">จำนวน</th>
      <th style="width:60px">ราคา</th>
      <th style="width:60px">รวม</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <hr class="divider">
  <div class="totals">
    {subtotal_row}{vat_row}
    <div class="t-row grand"><span>ยอดรวมทั้งหมด :</span><span>{total_val:,.2f} บาท</span></div>
  </div>
</div>
<script>
function reportHeight(){{
  var h=document.getElementById('card').getBoundingClientRect().height+8;
  window.parent.postMessage({{type:'recaipt_card_height',height:h}},'*');
}}
if(document.readyState==='complete'){{reportHeight();}}
else{{window.addEventListener('load',reportHeight);}}
if(typeof ResizeObserver!=='undefined'){{new ResizeObserver(reportHeight).observe(document.getElementById('card'));}}
</script>
</body></html>"""

# =========================================================
# PAGE ROUTING
# =========================================================
if "app_phase" not in st.session_state:
    st.session_state["app_phase"] = "UPLOAD"

# ── Shared header HTML ──
def render_header(badge_text="✨ Typhoon OCR"):
    st.markdown(f"""
    <div class="app-header">
      <div class="app-logo">
        <div class="app-logo-icon">📄</div>
        <div>
          <div class="app-logo-text">RecAipt</div>
          <div class="app-logo-sub">OCR Document Categorizer</div>
        </div>
      </div>
      <div class="header-badge">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PHASE 1 : UPLOAD
# ---------------------------------------------------------
if st.session_state["app_phase"] == "UPLOAD":
    render_header("✨ Typhoon OCR")
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "", type=["jpg", "jpeg", "png", "pdf"],
        key="uploader_widget", label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        with st.spinner("⏳ กำลังวิเคราะห์และหาขอบเอกสาร..."):
            img = load_image_or_pdf(file_bytes, file_name)
            if img is None:
                st.error("❌ ไม่รองรับไฟล์ประเภทนี้")
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
            st.error("❌ OCR ล้มเหลว กรุณาลองใหม่")
            if st.button("🔄 ลองใหม่"):
                reset_app()
                st.rerun()
        else:
            with st.spinner("🤖 กำลังจัดหมวดหมู่ข้อมูล..."):
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

    render_header("✅ สแกนเสร็จสมบูรณ์")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="img-card-wrap">', unsafe_allow_html=True)
        display_img = (
            cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            if len(processed_img.shape) == 2
            else cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        )
        st.image(display_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("← กลับหน้าหลัก", key="back_btn", type="secondary"):
            reset_app()
            st.rerun()

    with col_right:
        if has_error:
            st.error(f"❌ {extracted_json['error']}")
        else:
            items_count = len(extracted_json.get("items", []) or [])
            subtotal_val = safe_float(extracted_json.get("amount_before_tax", extracted_json.get("subtotal", 0)))
            vat_val = safe_float(extracted_json.get("vat_amount", extracted_json.get("vat", 0)))
            totals_h = 52 + (18 if subtotal_val else 0) + (18 if vat_val else 0)
            card_height = 44 + 35 + 30 + 114 + 58 + 28 + 30 + max(1, items_count) * 42 + totals_h + 24
            components.html(build_detail_card_html(extracted_json), height=card_height, scrolling=False)