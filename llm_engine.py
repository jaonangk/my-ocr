import json
import re
import requests
import streamlit as st

_THAI_CORRECTIONS = {
    "ภาษีมูลค่าเพิน": "ภาษีมูลค่าเพิ่ม",
    "ภาษีมูลคาเพิม": "ภาษีมูลค่าเพิ่ม",
    "ยอดรวมทงหมด": "ยอดรวมทั้งหมด",
    "ยอดรวมทงัหมด": "ยอดรวมทั้งหมด",
    "ใบเสรจ": "ใบเสร็จ",
    "ใบเสรจรบเงน": "ใบเสร็จรับเงิน",
    "เลขทใบ": "เลขที่ใบ",
    "เลขท ": "เลขที่ ",
    "วนท": "วันที่",
    "วนที": "วันที่",
    "เลขผเู้สย": "เลขผู้เสียภาษี",
    "เลขผู้เสยี": "เลขผู้เสียภาษี",
    "จำนวนเงน": "จำนวนเงิน",
    "รวมเปน": "รวมเป็น",
    "มูลคา": "มูลค่า",
    "ราคา/หนวย": "ราคา/หน่วย",
    "จำนวน/หนวย": "จำนวน/หน่วย",
}

def rule_based_correct(text):
    for wrong, correct in _THAI_CORRECTIONS.items():
        text = text.replace(wrong, correct)

    lines = text.split("\n")
    corrected_lines = []
    for line in lines:
        digit_count = sum(1 for char in line if char.isdigit())
        if len(line) > 0 and (digit_count / max(len(line), 1)) > 0.3:
            line = re.sub(r"\bO\b", "0", line)
            line = re.sub(r"\bl\b", "1", line)
        corrected_lines.append(line)
    return "\n".join(corrected_lines)

def clean_and_format_ocr(text):
    text = re.sub(r"```[a-zA-Z]*\n", "", text)
    text = text.replace("\n```", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("**", "")

    keywords = [
        r"(POS No\.?)", r"(Term\.?\s*No\.?)", r"(Staff\s*:?)", r"(Cashier\s*:?)",
        r"(Date\s*:?)", r"(Time\s*:?)", r"(TAX ID\s*:?)", r"(Rec No\.?)",
        r"(No\.CAS)", r"(Description Amount)", r"(รายการ)", r"(ราคาต่อหน่วย)",
        r"(?<!\n)(\b\d+\s*\))", r"(?<!\n)(\b\d+\.\s+[A-Za-zก-๙])", r"(1 EA @)",
        r"(Sub Total)", r"(Subtotal)", r"(Total\(VAT)", r"(Total Amount)",
        r"(ค่าพื้นที่ห่างไกล:?)", r"(ค่าส่ง:?)", r"(ส่วนลดรวม:?)", r"(ยอดรวม:?)",
        r"(ยอดสุทธิ:?)", r"(QR Promptpay)", r"(รับเงิน:?)", r"(เงินทอน:?)",
        r"(ขอบคุณ)", r"(สแกน QR)", r"(ผู้ส่ง)",
    ]
    for keyword in keywords:
        text = re.sub(keyword, r"\n\1", text, flags=re.IGNORECASE)

    lines = text.split("\n")
    formatted = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"^[-#*]+\s*", "", line)
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        formatted.append(line)
    return "\n".join(formatted)

def call_typhoon_llm(ocr_text):
    url = "https://api.opentyphoon.ai/v1/chat/completions"
    
    # 🔑 ดึง API Key อย่างปลอดภัยจาก Secrets ของ Streamlit
    api_key = st.secrets["OPENTYPHOON_API_KEY"]

    corrected_ocr = rule_based_correct(ocr_text)
    final_ocr_input = clean_and_format_ocr(corrected_ocr)

    system_prompt = (
        "คุณคือผู้เชี่ยวชาญอ่านเอกสารภาษาไทย/อังกฤษ "
        "หน้าที่คือสกัดข้อมูลตามบริบทลงใน JSON โครงสร้างตามที่กำหนดให้เท่านั้น "
        "ห้ามเดาข้อมูล ถ้าไม่มีข้อมูลในฟิลด์นั้นให้ใช้ค่า null และตอบกลับมาแค่ก้อน JSON เท่านั้น "
        "ห้ามเขียนอธิบาย ห้ามใส่กล่องข้อความ ```json ห้ามตอบอย่างอื่นเด็ดขาด"
    )

    user_prompt = f"""ข้อความ OCR:
----------------
{final_ocr_input}
----------------

สกัดเป็นโครงสร้าง JSON ดังนี้:
{{
  "store_name": "ชื่อหัวข้อ หรือ ชื่อร้าน หรือ null",
  "tax_id": "เลขผู้เสียภาษี หรือ null",
  "receipt_no": "เลขที่เอกสาร หรือ null",
  "date": "YYYY-MM-DD หรือ null",
  "items": [
    {{
      "name": "ชื่อรายการ/สินค้า",
      "qty": จำนวน,
      "unit_price": ราคา,
      "amount": ราคารวม
    }}
  ],
  "subtotal": ยอดก่อนภาษี หรือ null,
  "vat_rate": เปอร์เซ็นต์ VAT หรือ null,
  "vat": จำนวน VAT หรือ null,
  "total": ยอดรวมสุทธิ หรือ null,
  "payment_method": "วิธีชำระ หรือ null"
}}"""

    payload = {
        "model": "typhoon-v2.5-30b-a3b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        content = re.sub(r"\n```json", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```", "", content)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0).strip())
        return json.loads(content)
    except Exception as exc:
        return {"error": f"ไม่สามารถถอดโครงสร้างฟิลด์ได้: {str(exc)}"}