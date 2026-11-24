"""
BharatDoc AI - Extraction Prompt Templates
Preserves the exact research prompts, schemas, and 9 clean training demonstrations
used during Google Colab experimental evaluation.
"""

import json
from typing import Dict, Any, List
from src.config import CATEGORY_SCHEMAS

# Exact Research System Prompt (Phase 2.2 LoRA Fine-Tuning & Four-Way Comparison)
EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction system.\n"
    "Extract only information explicitly supported by the OCR text.\n"
    "Never guess or invent missing values.\n"
    "Return exactly the requested fields as valid JSON.\n"
    "Use an empty string when a field is missing or unreadable."
)

# Exact Nine Clean Training Demonstrations (3 Languages x 3 Categories)
# Derived from training split (train.jsonl) with clean EasyOCR transcription
NINE_FEW_SHOT_DEMOS = [
    {
        "language": "english",
        "category": "invoice",
        "document_id": "english_invoice_001",
        "ocr_text": "INVOICE\nCustomer Name: Kiran Rao\nInvoice Date: 01/01/2026\nInvoice Number: INV-0001\nAddress: Bengaluru\nTotal Amount: 1637",
        "ground_truth": {
            "name": "Kiran Rao",
            "date": "01/01/2026",
            "document_id": "INV-0001",
            "address": "Bengaluru",
            "amount": "1637"
        }
    },
    {
        "language": "english",
        "category": "application",
        "document_id": "english_application_011",
        "ocr_text": "APPLICATION FORM\nApplicant Name: Sneha Nair\nApplication Date: 11/01/2026\nApplication Number: APP-0011\nAddress: Bengaluru\nPhone Number: 9000001507",
        "ground_truth": {
            "name": "Sneha Nair",
            "date": "11/01/2026",
            "document_id": "APP-0011",
            "address": "Bengaluru",
            "phone": "9000001507"
        }
    },
    {
        "language": "english",
        "category": "certificate",
        "document_id": "english_certificate_021",
        "ocr_text": "CERTIFICATE OF COMPLETION\nCandidate Name: Priya Kumar\nIssue Date: 21/01/2026\nCertificate Number: CERT-0021\nInstitution: National Institute of Technology\nGrade: A",
        "ground_truth": {
            "name": "Priya Kumar",
            "date": "21/01/2026",
            "document_id": "CERT-0021",
            "institution": "National Institute of Technology",
            "grade": "A"
        }
    },
    {
        "language": "hindi",
        "category": "invoice",
        "document_id": "hindi_invoice_031",
        "ocr_text": "चालान\nग्राहक का नाम: मीरा रेड्डी\nचालान दिनांक: 31/01/2026\nचालान संख्या: INV-0031\nपता: बेंगलुरु\nकुल राशि: 5747",
        "ground_truth": {
            "name": "मीरा रेड्डी",
            "date": "31/01/2026",
            "document_id": "INV-0031",
            "address": "बेंगलुरु",
            "amount": "5747"
        }
    },
    {
        "language": "hindi",
        "category": "application",
        "document_id": "hindi_application_041",
        "ocr_text": "आवेदन पत्र\nआवेदक का नाम: विक्रम जोशी\nआवेदन दिनांक: 10/02/2026\nआवेदन संख्या: APP-0041\nपता: बेंगलुरु\nफोन नंबर: 9000005617",
        "ground_truth": {
            "name": "विक्रम जोशी",
            "date": "10/02/2026",
            "document_id": "APP-0041",
            "address": "बेंगलुरु",
            "phone": "9000005617"
        }
    },
    {
        "language": "hindi",
        "category": "certificate",
        "document_id": "hindi_certificate_051",
        "ocr_text": "पूर्णता प्रमाण पत्र\nअभ्यर्थी का नाम: किरण रेड्डी\nजारी दिनांक: 20/02/2026\nप्रमाण पत्र संख्या: CERT-0051\nसंस्थान: राष्ट्रीय प्रौद्योगिकी संस्थान\nश्रेणी: A",
        "ground_truth": {
            "name": "किरण रेड्डी",
            "date": "20/02/2026",
            "document_id": "CERT-0051",
            "institution": "राष्ट्रीय प्रौद्योगिकी संस्थान",
            "grade": "A"
        }
    },
    {
        "language": "kannada",
        "category": "invoice",
        "document_id": "kannada_invoice_061",
        "ocr_text": "ಸರಕುಪಟ್ಟಿ\nಗ್ರಾಹಕರ ಹೆಸರು: ಕಿರಣ್ ಮೆನನ್\nದಿನಾಂಕ: 02/03/2026\nಸರಕುಪಟ್ಟಿ ಸಂಖ್ಯೆ: INV-0061\nವಿಳಾಸ: ಬೆಂಗಳೂರು\nಒಟ್ಟು ಮೊತ್ತ: 9857",
        "ground_truth": {
            "name": "ಕಿರಣ್ ಮೆನನ್",
            "date": "02/03/2026",
            "document_id": "INV-0061",
            "address": "ಬೆಂಗಳೂರು",
            "amount": "9857"
        }
    },
    {
        "language": "kannada",
        "category": "application",
        "document_id": "kannada_application_071",
        "ocr_text": "ಅರ್ಜಿ ನಮೂನೆ\nಅರ್ಜಿದಾರರ ಹೆಸರು: ಅರ್ಜುನ್ ನಾಯರ್\nಅರ್ಜಿ ದಿನಾಂಕ: 12/03/2026\nಅರ್ಜಿ ಸಂಖ್ಯೆ: APP-0071\nವಿಳಾಸ: ಬೆಂಗಳೂರು\nದೂರವಾಣಿ ಸಂಖ್ಯೆ: 9000009727",
        "ground_truth": {
            "name": "ಅರ್ಜುನ್ ನಾಯರ್",
            "date": "12/03/2026",
            "document_id": "APP-0071",
            "address": "ಬೆಂಗಳೂರು",
            "phone": "9000009727"
        }
    },
    {
        "language": "kannada",
        "category": "certificate",
        "document_id": "kannada_certificate_081",
        "ocr_text": "ಪೂರ್ಣಗೊಂಡ ಪ್ರಮಾಣಪತ್ರ\nಅಭ್ಯರ್ಥಿಯ ಹೆಸರು: ಮೀರಾ ಶರ್ಮಾ\nವಿತರಣಾ ದಿನಾಂಕ: 22/03/2026\nಪ್ರಮಾಣಪತ್ರ ಸಂಖ್ಯೆ: CERT-0081\nಸಂಸ್ಥೆ: ರಾಷ್ಟ್ರೀಯ ತಂತ್ರಜ್ಞಾನ ಸಂಸ್ಥೆ\nಶ್ರೇಣಿ: A",
        "ground_truth": {
            "name": "ಮೀರಾ ಶರ್ಮಾ",
            "date": "22/03/2026",
            "document_id": "CERT-0081",
            "institution": "ರಾಷ್ಟ್ರೀಯ ತಂತ್ರಜ್ಞಾನ ಸಂಸ್ಥೆ",
            "grade": "A"
        }
    }
]

def get_few_shot_demonstration(language: str, category: str) -> Dict[str, Any]:
    """Retrieve the exact demonstration record matching language and category."""
    lang = language.lower()
    cat = category.lower()
    for demo in NINE_FEW_SHOT_DEMOS:
        if demo["language"] == lang and demo["category"] == cat:
            return demo
    return NINE_FEW_SHOT_DEMOS[0]

def build_user_prompt(
    ocr_text: str,
    category: str,
    language: str,
    few_shot: bool = False
) -> str:
    """
    Construct user prompt matching Cell 28 of Phase 2.2 research notebook.
    Supports both zero-shot and few-shot in-context demonstration formats.
    """
    cat = category.lower()
    lang = language.lower()
    fields = CATEGORY_SCHEMAS.get(cat, [])
    
    prompt = ""
    if few_shot:
        prompt += "Here are examples of correct information extraction:\n\n"
        # Use the matched domain demonstration
        demo = get_few_shot_demonstration(lang, cat)
        demo_fields = CATEGORY_SCHEMAS.get(demo["category"], [])
        demo_output = {f: demo["ground_truth"].get(f, "") for f in demo_fields}
        
        prompt += (
            f"Example language: {demo['language']}\n"
            f"Example category: {demo['category']}\n"
            f"OCR Text:\n{demo['ocr_text']}\n\n"
            f"Correct JSON:\n"
            f"{json.dumps(demo_output, ensure_ascii=False, indent=2)}\n\n"
        )
        
    prompt += (
        f"Language: {lang}\n"
        f"Category: {cat}\n"
        f"OCR Text:\n{ocr_text}\n\n"
        f"Extract these fields: {', '.join(fields)}"
    )
    
    return prompt.strip()
