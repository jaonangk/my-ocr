import cv2
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
    crop_document_contour, # เพิ่มฟังก์ชันตัดขอบเข้ามา
    deskew_image,
    load_image_or_pdf,
    process_method_4_sharpening,
    run_typhoon_ocr,
)

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

.header-bar {
    display:flex; justify-content:space-between; align-items:center;
    padding:14px 28px; margin-bottom:40px;
    background:#FFFFFF; border-radius:18px;
    box-shadow:0 4px 15px rgba(74,46,53,0.02);
}
.logo-text {
    color:#4A2E35; font-size:20px; font-weight:700;
    display:flex; align-items:center; gap:8px;
}
.lang-pill {
    background:#C97D98; color:white;
    padding:7px 16px; border-radius:10px;
    font-size:13px; font-weight:500;
}
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
    align-items:center holiday;
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

import re

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
# 🔥 JAVASCRIPT MESSAGE TUNNEL
# =========================================================
st.markdown("""
<script>
if (!window.hasRecAiptListener) {
    window.hasRecAiptListener = true;
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'recalpt_action') {
            const url = new URL(window.parent.location.href);
            url.searchParams.set("triggered_event", e.data.value);
            window.parent.location.href = url.toString();
        }
    });
}
</script>
""", unsafe_allow_html=True)

HTML_POST_BRIDGE = """<script>
function executeAction(actValue) {
    window.parent.postMessage({type: 'recalpt_action', value: actValue}, '*');
}
</script>"""

# =========================================================
# HTML BUILDERS & SVG ICONS
# =========================================================
SVG_BACK = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>'
SVG_EDIT = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>'
SVG_DELETE = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>'
SVG_COPY = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
SVG_SHARE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>'
SVG_DL = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
SVG_ZOOM = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>'
SVG_BOX = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C97D98" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>'

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
.card{{background:#FFF6F8;border-radius:24px;border:1px solid #F8D7E3;padding:24px 28px;overflow:hidden}}
.dc-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}}
.dc-back{{width:32px;height:32px;border-radius:50%;background:#F8D7E3;color:#A35271;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .15s}}
.dc-back:hover{{background:#F4C6D5}}
.dc-title{{font-size:17px;font-weight:700;color:#4A2E35;flex:1;text-align:center}}
.dc-icons{{display:flex;gap:8px;flex-shrink:0}}
.icon-btn{{width:32px;height:32px;border-radius:8px;background:#FFF0F5;color:#A35271;border:1px solid #F4C6D5;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s}}
.icon-btn:hover{{background:#F4C6D5}}
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
{HTML_POST_BRIDGE}
</head><body>
<div class="card">
  <div class="dc-header">
    <button class="dc-back" onclick="executeAction('back')">{SVG_BACK}</button>
    <span class="dc-title">รายละเอียดเอกสาร (OCR Categorized)</span>
    <div class="dc-icons">
      <button class="icon-btn" title="แก้ไข" onclick="executeAction('edit')">{SVG_EDIT}</button>
      <button class="icon-btn" title="ลบ" onclick="executeAction('delete')">{SVG_DELETE}</button>
    </div>
  </div>
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

def build_action_bar_html():
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent}}
svg{{display:inline-block;vertical-align:middle}}
.bar{{display:flex;gap:10px;width:100%;padding:2px}}
.btn{{flex:1;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;gap:7px;font-size:14px;font-weight:600;cursor:pointer;border:1px solid #F4C6D5;background:#FFF0F5;color:#A35271;transition:background .15s,transform .1s;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
.btn:hover{{background:#F4C6D5;transform:translateY(-1px)}}
.btn:active{{transform:translateY(0)}}
.btn.primary{{background:#C97D98;color:#fff;border:none}}
.btn.primary:hover{{background:#A35271}}
</style>
{HTML_POST_BRIDGE}
</head><body>
<div class="bar">
  <button class="btn" onclick="executeAction('copy')">{SVG_COPY} คัดลอก</button>
  <button class="btn" onclick="executeAction('share')">{SVG_SHARE} แชร์</button>
  <button class="btn primary" onclick="executeAction('export')">{SVG_DL} ส่งออก (Save)</button>
</div>
</body></html>"""

def build_img_controls_html():
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent}}
svg{{display:inline-block;vertical-align:middle}}
.row{{display:flex;justify-content:space-between;align-items:center;padding:10px 2px 2px}}
.round-btn{{width:42px;height:42px;border-radius:50%;background:#FFFFFF;border:1px solid #F4C6D5;cursor:pointer;box-shadow:0 2px 8px rgba(74,46,53,0.08);color:#4A2E35;transition:background .15s}}
.round-btn:hover{{background:#F8D7E3}}
</style>
{HTML_POST_BRIDGE}
</head><body>
<div class="row">
  <button class="round-btn" onclick="executeAction('back')">{SVG_BACK}</button>
  <button class="round-btn" onclick="executeAction('maximize')">{SVG_ZOOM}</button>
</div>
</body></html>"""

# =========================================================
# PAGE 1 : UPLOAD
# =========================================================
if "processed_img" not in st.session_state or st.session_state.get("file_uploaded") is None:
    st.markdown("<div class='hero-title'>OCR Document Categorizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>ถ่ายภาพเอกสารมุมกว้าง ระบบจะ Crop ขอบ และสกัดข้อความจัดหมวดหมู่ให้อัตโนมัติ</div>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"], key="uploader_widget",
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        st.session_state["file_uploaded"] = uploaded_file
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        with st.spinner("⏳ ตัดขอบเอกสาร (Smart Cropping) และประมวลผล..."):
            img = load_image_or_pdf(file_bytes, file_name)
            if img is None: st.error("❌ Unsupported file"); st.stop()
            
            # 1. ตัดรูปสิ่งของรอบๆ ออกให้เหลือเฉพาะเอกสาร (ตามโจทย์เป๊ะ)
            cropped_img = crop_document_contour(img)
            
            # 2. แก้เอียงและเพิ่มความชัด
            deskewed = deskew_image(cropped_img)
            processed = process_method_4_sharpening(deskewed)
            st.session_state["processed_img"] = processed

        with st.spinner("⚡ กำลังอ่านข้อความด้วย OCR..."):
            raw_text = run_typhoon_ocr(st.session_state["processed_img"])
            st.session_state["raw_text"] = raw_text

        if "[ERROR]" in raw_text or not raw_text.strip():
            st.error("❌ OCR failed");
            st.session_state.clear()
        else:
            with st.spinner("🤖 กำลังจัดหมวดหมู่ข้อมูล (Categorizing)..."):
                extracted_json = call_typhoon_llm(raw_text)
                st.session_state["extracted_json"] = extracted_json
            st.rerun()

# =========================================================
# PAGE 2 : RESULT DASHBOARD (Sandbox Mode)
# =========================================================
else:
    processed_img = st.session_state["processed_img"]
    raw_text = st.session_state["raw_text"]
    extracted_json = st.session_state["extracted_json"]

    has_error = (isinstance(extracted_json, dict) and "error" in extracted_json and extracted_json["error"])
    action = st.query_params.get("triggered_event", "")

    if action == "back":
        st.query_params.clear()
        reset_app()
        st.rerun()
    elif action == "edit":
        st.query_params.clear();
        st.toast("✏️ ระบบ: เปิดสิทธิ์ให้แก้ไขข้อความที่ OCR อ่านได้ (Editable Text)")
    elif action == "delete":
        st.query_params.clear();
        st.toast("🗑️ ระบบ: ล้างข้อมูลจำลองบนหน้าจอเรียบร้อย")
    elif action == "copy":
        st.query_params.clear();
        st.toast("📋 ระบบ: คัดลอกข้อมูลลง Clipboard")
    elif action == "share":
        st.query_params.clear();
        st.toast("🔗 ระบบ: คัดลอกลิงก์แชร์เอกสารสำเร็จ")
    elif action == "maximize":
        st.query_params.clear();
        st.toast("🔍 ระบบ: ขยายภาพเอกสารเต็มหน้าจอ")

    elif action == "export":
        st.query_params.clear()
        # เปลี่ยนเป็นโชว์ Success แบบ Sandbox Mode แทนการบันทึกลง DB
        st.success(f"🎉 ประมวลผลและแยกหมวดหมู่เอกสารเสร็จสมบูรณ์ (พร้อมนำไปใช้งานต่อ)")
        st.balloons()

    st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="img-card-wrap">', unsafe_allow_html=True)
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB) if len(
            processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        components.html(build_img_controls_html(), height=58, scrolling=False)

    with col_right:
        if has_error:
            st.error(f"❌ {extracted_json['error']}")
        else:
            items_count = len(extracted_json.get("items", []) or [])
            card_height = 430 + max(0, items_count - 2) * 38
            components.html(build_detail_card_html(extracted_json), height=card_height, scrolling=False)
            components.html(build_action_bar_html(), height=58, scrolling=False)

    st.markdown('</div>', unsafe_allow_html=True)