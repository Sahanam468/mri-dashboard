# =============================================================
#  AI MRI Dashboard: Detection → Segmentation → Gemini LLM → PDF
# =============================================================

import os
import re
import cv2
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image as PILImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import qrcode


# =============================================================
# ENV + API KEY
# =============================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

HAVE_GENAI = False
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        HAVE_GENAI = True
    except:
        HAVE_GENAI = False


# =============================================================
# PATH CONFIG  (YOUR WINDOWS PATHS)
# =============================================================
DETECTION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_inceptionv3_tumor.h5")
SEGMENTATION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tumor_segmentation_unet.h5")
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
PDF_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")

os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)


# =============================================================
# GEMINI FUNCTION (SAFE FALLBACK)
# =============================================================
def gemini_generate(prompt: str):
    """Generate text using Gemini with correct SDK. Safe fallback if it fails."""
    if not HAVE_GENAI:
        return (
            "Findings: LLM failed.\n\n"
            "Impression: Manual review required.\n\n"
            "Recommendations: Review clinically."
        )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return (
            "Findings: LLM failed.\n\n"
            "Impression: Manual review required.\n\n"
            "Recommendations: Review clinically."
        )



# =============================================================
# MODEL LOADING (CACHED)
# =============================================================
@st.cache_resource
def load_detection(path):
    return load_model(path)


@st.cache_resource
def load_segmentation(path):
    return load_model(path, compile=False)


detection_model = load_detection(DETECTION_MODEL_PATH)
segmentation_model = load_segmentation(SEGMENTATION_MODEL_PATH)

CLASS_LABELS = ["glioma", "meningioma", "no_tumor", "pituitary"]


# =============================================================
# MRI ANALYSIS — FINAL CORRECTED FUNCTION
# =============================================================
def analyze_mri_local(img_path: str):

    # ------------------ DETECTION ------------------
    img = image.load_img(img_path, target_size=(224, 224))
    arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    preds = detection_model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)

    label = CLASS_LABELS[idx]
    confidence = float(preds[idx] * 100)

    mask_path = None
    tumor_pct = 0.0

    # ------------------ SEGMENTATION ------------------
    if label != "no_tumor" and segmentation_model is not None:

        seg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        seg = cv2.equalizeHist(seg)
        seg = cv2.resize(seg, (128, 128))

        seg_arr = seg.astype("float32") / 255.0
        seg_arr = np.expand_dims(seg_arr, axis=-1)
        seg_arr = np.expand_dims(seg_arr, axis=0)

        mask_pred = segmentation_model.predict(seg_arr, verbose=0)[0]

        mask = mask_pred[:, :, 0] if mask_pred.ndim == 3 else mask_pred

        # Lower threshold helps your model
        binary = (mask > 0.15).astype(np.uint8)

        tumor_pct = float(binary.sum() / binary.size * 100)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = os.path.join(PDF_OUTPUT_DIR, f"mask_{timestamp}.png")
        plt.imsave(mask_path, binary * 255, cmap="gray")

    return {
        "image_path": img_path,
        "predicted_label": label,
        "confidence": confidence,
        "tumor_percentage": tumor_pct,
        "mask_image_path": mask_path
    }


# =============================================================
# PDF GENERATOR
# =============================================================
def clean_text_for_pdf(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def make_qr(data, out):
    qr = qrcode.QRCode(box_size=3, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").convert("RGB").save(out)
    return out


def create_apollo_style_pdf(report_text, original_img, mask_img, patient, pid, apt_id):

    report_text = clean_text_for_pdf(report_text)
    study_date = datetime.now().strftime("%d-%b-%Y %I:%M %p")

    output_path = os.path.join(
        PDF_OUTPUT_DIR,
        f"MRI_Report_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    qr_path = os.path.join(PDF_OUTPUT_DIR, f"QR_{pid}.png")
    make_qr(f"{patient}", qr_path)

    doc = SimpleDocTemplate(output_path, pagesize=A4)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Heading1'], alignment=1,
                                 fontSize=16, textColor=colors.HexColor("#0D47A1"))
    section_style = ParagraphStyle('section', parent=styles['Heading2'],
                                   textColor=colors.HexColor("#0D47A1"))

    body_style = ParagraphStyle('body', parent=styles['BodyText'], fontSize=10)

    story = []

    story.append(Paragraph("<b>AI Diagnostic Center</b>", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("MRI Brain Tumor Report", section_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Patient:</b> {patient['name']}", body_style))
    story.append(Paragraph(f"<b>Age:</b> {patient['age']}    <b>Gender:</b> {patient['gender']}", body_style))
    story.append(Paragraph(f"<b>PID:</b> {pid}   <b>Date:</b> {study_date}", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Findings</b>", section_style))
    story.append(Paragraph(report_text.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 12))

    imgs = []
    if os.path.exists(original_img):
        imgs.append(RLImage(original_img, width=220, height=220))
    if mask_img and os.path.exists(mask_img):
        imgs.append(RLImage(mask_img, width=220, height=220))

    story.append(Table([imgs]))
    doc.build(story)

    return output_path


# =============================================================
# STREAMLIT UI
# =============================================================
st.set_page_config(page_title="AI MRI Brain Tumor Diagnostic", layout="wide")

# Session state
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {}
if "last_report" not in st.session_state:
    st.session_state.last_report = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
    page = st.radio("Navigate", ["Home", "Analyze", "Assistant", "About"], index=1)


# ====================== HOME ======================
# ====================== HOME ======================
if page == "Home":
    st.markdown("<h1 class='title-w'>AI MRI Brain Tumor Diagnostic</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p class='subtitle-w'>Upload MRI → Automated Detection → Segmentation → PDF → Ask AI Assistant</p>",
        unsafe_allow_html=True
    )

    # ----------- Animated Image (Professional MRI Scan GIF) -----------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <div style='text-align:center; margin-bottom:20px;'>
        <img src='https://i.gifer.com/NTHO.gif' max-width=60% height='400'>
    
    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation button → Analyze page
    if st.button("🔍 Start Analysis"):
        st.session_state['navigate_to_analyze'] = True

    # Auto-switch page when button is clicked
    if st.session_state.get("navigate_to_analyze"):
        page = "Analyze"




# ====================== ANALYZE ======================
elif page == "Analyze":

    st.header("Upload MRI → Run Detection & Segmentation")
    # Cute + professional MRI animation
    st.markdown(
    """
    <div style='text-align:center; margin-bottom:10px;'>
        <img src='https://i.gifer.com/HesE.gif' width='450' height='450'>
    
    </div>
    """,
    unsafe_allow_html=True
    )
  


    upload = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if upload:
        temp_path = os.path.join(PDF_OUTPUT_DIR, f"input_{datetime.now().strftime('%H%M%S')}.jpg")
        with open(temp_path, "wb") as f:
            f.write(upload.getbuffer())

        st.image(temp_path, caption="Uploaded MRI", use_container_width=True)

        if st.button("Run Analysis"):

            with st.spinner("Running models..."):
                try:
                    result = analyze_mri_local(temp_path)
                    st.session_state.last_analysis = result
                    st.success("Analysis Complete ✔" )
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
                    result = None

        if st.session_state.last_analysis:
            r = st.session_state.last_analysis

            st.subheader("Results:")
            st.write(f"**Tumor Type:** {r['predicted_label'].upper()}")
            st.write(f"**Confidence:** {r['confidence']:.2f}%")
            st.write(f"**Tumor Coverage:** {r['tumor_percentage']:.2f}%")

            if r["mask_image_path"]:
                st.image(r["mask_image_path"], caption="Segmentation Mask")


            st.subheader("Generate PDF Report")
            name = st.text_input("Patient Name", "John Doe")
            age = st.number_input("Age", 0, 120, 45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

            if st.button("Generate Report"):
                prompt = f"""
Patient: {name}, Age: {age}, Gender: {gender}
Tumor Type: {r['predicted_label']}
Confidence: {r['confidence']:.2f}%
Coverage: {r['tumor_percentage']:.2f}%

Write a radiology report with Findings, Impression, Recommendations.
                """

                with st.spinner("Generating LLM Report..."):
                    report_text = gemini_generate(prompt)
                    st.session_state.last_report = report_text

                pdf = create_apollo_style_pdf(
                    report_text,
                    r["image_path"],
                    r["mask_image_path"],
                    {"name": name, "age": age, "gender": gender},
                    pid="PID" + datetime.now().strftime("%H%M%S"),
                    apt_id="APT001"
                )

                with open(pdf, "rb") as f:
                    st.download_button("Download PDF", f, file_name=os.path.basename(pdf))
                # Navigation buttons section
                st.markdown("<br><hr><br>", unsafe_allow_html=True)

                colA, colB = st.columns(2)

                with colA:
                 if st.button("🏠 Go to Home"):
                  st.session_state['navigate_to_home'] = True

                with colB:
                 if st.button("🤖 Go to Assistant"):
                  st.session_state['navigate_to_assistant'] = True

# Redirect logic
                 if st.session_state.get("navigate_to_home"):
                  page = "Home"

                 if st.session_state.get("navigate_to_assistant"):
                  page = "Assistant"
    




# ====================== ASSISTANT ======================
elif page == "Assistant":
    st.header("AI Assistant")
    st.markdown(
    """
    <div style='text-align:center; margin-bottom:20px;'>
        <img src='https://i.gifer.com/NTHO.gif' max-width=60% height='400'>
    
    </div>
    """,
    unsafe_allow_html=True
    )


    if not st.session_state.last_report:
        st.info("Run an analysis and generate a report first.")
    else:
        question = st.text_input("Ask anything about the findings")

        if st.button("Ask"):
            q = question.strip()
            if q:
                reply = gemini_generate(
                    f"Report:\n{st.session_state.last_report}\n\nUser: {q}\nAnswer safely and simply."
                )
                st.session_state.chat_history.append((q, reply))

        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")


# ====================== ABOUT ======================
elif page == "About":
    st.title("About This System")
    st.write("""


This system is an **AI-driven MRI Brain Tumor Analysis platform** designed for research and academic use.  
It automates tumor detection, segmentation, and report generation to support radiologists.

**Key Features**
- **Tumor Classification** (Glioma, Meningioma, Pituitary, No Tumor)
- **Tumor Segmentation** using U-Net  
- **Tumor Coverage Calculation**
- **AI-Generated Report** using Google Gemini  
- **Professional PDF Report** with images & metadata  
- **Interactive Assistant** for follow-up questions  

**Purpose:**  
To demonstrate how AI can assist in MRI interpretation.  
*Not a replacement for clinical diagnosis.*
""")

st.markdown("**Notes:**\n- This is a research/demo tool. Always confirm with clinical specialist.")

# End of app.py






# # === MRI → GEMINI LLM → APOLLO PDF (VS CODE READY VERSION) ===

# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from PIL import Image as PILImage

# # ============================
# # 0. IMPORT GEMINI SDK
# # ============================
# import google.generativeai as genai

# GEMINI_API_KEY = "AIzaSyDU7khj1hvqLRsME-smi5A5RLJmObvtB-M"
# genai.configure(api_key=GEMINI_API_KEY)
# llm_model = genai.GenerativeModel("gemini-2.5-flash")

# def gemini_generate_llm(prompt: str):
#     response = llm_model.generate_content(prompt)
#     return response.text


# # ============================
# # 1. MODEL PATHS (WINDOWS FIXED)
# # ============================
# DETECTION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\best_inceptionv3_tumor.h5"
# SEGMENTATION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\tumor_segmentation_unet.h5"

# OUTPUT_FOLDER = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\downloads"


# # ============================
# # 2. LOAD MODELS (TensorFlow)
# # ============================
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# print("📌 Loading models...")

# try:
#     detection_model = load_model(DETECTION_MODEL_PATH)
#     print("✔ Detection model loaded")
# except Exception as e:
#     print("❌ Detection model error:", e)
#     exit()

# try:
#     segmentation_model = load_model(SEGMENTATION_MODEL_PATH, compile=False)
#     print("✔ Segmentation model loaded")
# except Exception as e:
#     print("❌ Segmentation model error:", e)
#     exit()

# class_labels = ["glioma", "meningioma", "no_tumor", "pituitary"]


# # ============================
# # 3. MRI ANALYSIS
# # ============================
# def analyze_mri(img_path):

#     print(f"\n🧠 MRI Analysis Started: {img_path}")

#     img = image.load_img(img_path, target_size=(224, 224))
#     arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

#     preds = detection_model.predict(arr)[0]
#     idx = np.argmax(preds)

#     label = class_labels[idx]
#     conf = float(preds[idx] * 100)

#     print(f"🔍 Detected: {label} ({conf:.2f}%)")

#     # Segmentation
#     mask_path = None
#     tumor_pct = 0.0

#     if label != "no_tumor":
#         seg_img = image.load_img(img_path, color_mode="grayscale", target_size=(128, 128))
#         seg_arr = np.expand_dims(image.img_to_array(seg_img) / 255.0, axis=0)

#         mask = segmentation_model.predict(seg_arr)[0][:, :, 0]

#         mask_path = img_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")

#         plt.imsave(mask_path, mask, cmap="gray")

#         binary = (mask > 0.5).astype(np.uint8)
#         tumor_pct = float((binary.sum() / binary.size) * 100)

#     return {
#         "image_path": img_path,
#         "predicted_label": label,
#         "confidence": conf,
#         "tumor_percentage": tumor_pct,
#         "mask_image_path": mask_path
#     }


# # ============================
# # 4. PROMPT FOR GEMINI
# # ============================
# REPORT_PROMPT = """
# You are an expert radiology assistant. Generate a clean, professional MRI brain tumor radiology report.

# Patient Name: {patient_name}
# Age: {age}
# Gender: {gender}
# Date of Study: {date}

# Tumor Detection:
# - Type: {label}
# - Confidence: {confidence:.2f}%
# - Tumor Coverage: {tumor_percentage:.2f}%

# Provide structured sections:
# Findings:
# Impression:
# Recommendations:
# """

# def generate_report(data, patient):
#     today = datetime.now().strftime("%d %B %Y")

#     prompt = REPORT_PROMPT.format(
#         patient_name=patient["name"],
#         age=patient["age"],
#         gender=patient["gender"],
#         date=today,
#         label=data["predicted_label"],
#         confidence=data["confidence"],
#         tumor_percentage=data["tumor_percentage"]
#     )

#     return gemini_generate_llm(prompt)


# # ============================
# # 5. PDF GENERATION
# # ============================
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table, TableStyle
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors
# import qrcode

# def clean_text_for_pdf(text):
#     return re.sub(r"\*\*(.*?)\*\*", r"\1", text).strip()

# def make_qr(text, path):
#     qr = qrcode.QRCode(box_size=3, border=1)
#     qr.add_data(text)
#     qr.make(True)
#     qr.make_image().save(path)
#     return path

# def create_apollo_pdf(report_text, original_img, mask_img, patient, output_path):

#     report_text = clean_text_for_pdf(report_text)

#     doc = SimpleDocTemplate(output_path, pagesize=A4)

#     styles = getSampleStyleSheet()
#     title = ParagraphStyle("T", parent=styles["Heading1"], alignment=1, textColor=colors.HexColor("#0D47A1"))

#     story = []

#     story.append(Paragraph("AI MRI Brain Tumor Report", title))
#     story.append(Spacer(1, 10))

#     story.append(Paragraph(f"Patient Name: {patient['name']}", styles["BodyText"]))
#     story.append(Paragraph(f"Age: {patient['age']}   Gender: {patient['gender']}", styles["BodyText"]))
#     story.append(Spacer(1, 10))

#     story.append(Paragraph(report_text.replace("\n", "<br/>"), styles["BodyText"]))
#     story.append(Spacer(1, 15))

#     imgs = []
#     imgs.append(RLImage(original_img, width=200, height=200))

#     if mask_img:
#         imgs.append(RLImage(mask_img, width=200, height=200))

#     story.append(Table([imgs]))
#     story.append(Spacer(1, 20))

#     story.append(Paragraph("<font size=8>This is an AI-generated radiology report.</font>", styles["BodyText"]))

#     doc.build(story)

#     print("\n📄 PDF saved to:", output_path)


# # ============================
# # 6. RUN FULL PIPELINE
# # ============================

# # 👉 CHANGE TEST IMAGE HERE
# test_image = r"C:\Users\RAKSHITHA SRINIVASA\Downloads\Te-gl_0010.jpg"

# data = analyze_mri(test_image)

# patient_info = {
#     "name": "John Doe",
#     "age": 45,
#     "gender": "Male"
# }

# print("\n🧾 Generating Radiology Report...")
# report = generate_report(data, patient_info)

# output_pdf = os.path.join(OUTPUT_FOLDER, "final_mri_report.pdf")

# create_apollo_pdf(
#     report_text=report,
#     original_img=data["image_path"],
#     mask_img=data["mask_image_path"],
#     patient=patient_info,
#     output_path=output_pdf
# )

# print("\n🎉 DONE! PDF Generated Successfully!")




# app.py
# Streamlit dashboard (VS Code / local) for:
# MRI -> Detection -> Segmentation -> Gemini LLM -> Apollo-style PDF
# - First page: logo + introduction
# - Second: Analysis (upload image, run models, view results, generate & download PDF)
# - Third: Assistant (ask questions about latest report; simple session memory)
#
# Paths are Windows-ready — edit constants below if needed.
# Run with: `streamlit run app.py`



# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import re
# import io
# import numpy as np
# import streamlit as st
# from datetime import datetime
# from PIL import Image
# from PIL import Image as PILImage
# import matplotlib.pyplot as plt
# import qrcode

# # TensorFlow / Keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # ReportLab for PDF
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors

# # Optional Gemini SDK (uses google-generativeai). If not available, fallback to a safe text.
# try:
#     import google.generativeai as genai
#     HAVE_GENAI = True
# except Exception:
#     HAVE_GENAI = False

# # -------------------------
# # CONFIG — EDIT IF NEEDED
# # -------------------------
# # Use raw strings (r"") to avoid escape issues on Windows.
# GEMINI_API_KEY = "AIzaSyDU7khj1hvqLRsME-smi5A5RLJmObvtB-M"  # Optional: put your Gemini key here if you want LLM reports & assistant
# DETECTION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\best_inceptionv3_tumor.h5"
# SEGMENTATION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\tumor_segmentation_unet.h5"
# LOGO_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\logo.png"  # optional
# SIGNATURE_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\signature.png"  # optional
# OUTPUT_DIR = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\downloads"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Model label set (must match training)
# CLASS_LABELS = ["glioma", "meningioma", "no_tumor", "pituitary"]

# # -------------------------
# # CONFIGURE LLM (Gemini)
# # -------------------------
# def configure_genai(api_key: str) -> bool:
#     if not HAVE_GENAI or not api_key:
#         return False
#     try:
#         genai.configure(api_key=api_key)
#         return True
#     except Exception:
#         return False

# GENAI_READY = configure_genai(GEMINI_API_KEY)

# def gemini_generate(prompt: str, model: str = "gemini-2.5-flash") -> str:
#     """
#     Call Gemini via google.generativeai if available & configured.
#     Fallback returns a conservative templated report.
#     """
#     if not GENAI_READY:
#         # Safe fallback short report
#         return (
#             "Findings:\n- Automated detection completed.\n\n"
#             "Impression:\n- Please correlate with clinical information.\n\n"
#             "Recommendations:\n- Refer to specialist for confirmation."
#         )
#     try:
#         resp = genai.models.generate(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_output_tokens=600,
#         )
#         # Support different response shapes
#         if hasattr(resp, "candidates") and resp.candidates:
#             # candidate content may be nested; handle common variations defensively
#             cand = resp.candidates[0]
#             if hasattr(cand, "content") and cand.content:
#                 # candidate.content is often a list of dicts
#                 # join textual parts
#                 parts = []
#                 for c in cand.content:
#                     if isinstance(c, dict) and "text" in c:
#                         parts.append(c["text"])
#                     elif isinstance(c, str):
#                         parts.append(c)
#                 return "\n".join(parts) if parts else str(cand)
#             # fallback string form
#             return str(cand)
#         if hasattr(resp, "content") and resp.content:
#             if isinstance(resp.content, list):
#                 # join text fields
#                 texts = []
#                 for item in resp.content:
#                     if isinstance(item, dict) and "text" in item:
#                         texts.append(item["text"])
#                     elif isinstance(item, str):
#                         texts.append(item)
#                 return "\n".join(texts)
#             return str(resp.content)
#         return str(resp)
#     except Exception as e:
#         return (
#             "Findings:\n- Automated detection completed (LLM failed).\n\n"
#             "Impression:\n- Please review manually.\n\n"
#             "Recommendations:\n- Specialist review recommended."
#         )

# # -------------------------
# # MODEL LOADING (cached)
# # -------------------------
# @st.cache_resource
# def load_detection(path: str):
#     try:
#         return load_model(path)
#     except Exception as e:
#         st.error(f"Detection model load error: {e}")
#         return None

# @st.cache_resource
# def load_segmentation(path: str):
#     try:
#         return load_model(path, compile=False)
#     except Exception as e:
#         # segmentation optional
#         return None

# detection_model = load_detection(DETECTION_MODEL_PATH)
# segmentation_model = load_segmentation(SEGMENTATION_MODEL_PATH)

# # -------------------------
# # ANALYSIS FUNCTION
# # -------------------------
# def analyze_mri_local(img_path: str):
#     """
#     Returns dict: image_path, predicted_label, confidence, tumor_percentage, mask_image_path
#     """
#     if not os.path.exists(img_path):
#         raise FileNotFoundError(f"Image not found: {img_path}")

#     # preprocess for classifier
#     img = image.load_img(img_path, target_size=(224, 224))
#     arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

#     # detection fallback
#     if detection_model:
#         preds = detection_model.predict(arr, verbose=0)[0]
#     else:
#         preds = np.array([0.0, 0.0, 1.0, 0.0])  # default: no_tumor

#     idx = int(np.argmax(preds))
#     label = CLASS_LABELS[idx]
#     confidence = float(preds[idx] * 100.0)

#     tumor_percentage = 0.0
#     mask_path = None

#     if label != "no_tumor" and segmentation_model:
#         # segmentation expects grayscale 128x128 in many setups
#         seg_img = image.load_img(img_path, color_mode='grayscale', target_size=(128, 128))
#         seg_arr = np.expand_dims(image.img_to_array(seg_img) / 255.0, axis=0)
#         mask = segmentation_model.predict(seg_arr, verbose=0)[0]
#         # ensure first channel
#         if mask.ndim == 3:
#             m = mask[:, :, 0]
#         else:
#             m = mask
#         # binary mask and save
#         binary = (m > 0.3).astype(np.uint8) * 255
#         base = os.path.splitext(img_path)[0]
#         mask_path = base + "_mask.png"
#         PILImage.fromarray(binary.astype(np.uint8)).save(mask_path)
#         tumor_percentage = float((binary.astype(np.uint8) > 0).sum() / (binary.size) * 100.0)
        

#     return {
#         "image_path": img_path,
#         "predicted_label": label,
#         "confidence": confidence,
#         "tumor_percentage": tumor_percentage,
#         "mask_image_path": mask_path
#     }

# # -------------------------
# # PDF GENERATOR (Apollo-style)
# # -------------------------
# def clean_text_for_pdf(text: str) -> str:
#     if not text:
#         return ""
#     text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
#     text = re.sub(r"\*(.*?)\*", r"\1", text)
#     return text.strip()

# def create_pdf(report_text: str, original_img: str, mask_img: str, patient: dict, pid="PID000", apt="APT000", output_path=None):
#     report_text = clean_text_for_pdf(report_text)
#     if output_path is None:
#         output_path = os.path.join(OUTPUT_DIR, f"mri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

#     # Build PDF
#     doc = SimpleDocTemplate(output_path, pagesize=A4,
#                             topMargin=18, bottomMargin=22, leftMargin=22, rightMargin=22)
#     styles = getSampleStyleSheet()
#     title_style = ParagraphStyle('title', parent=styles['Heading1'], alignment=1, fontSize=15, textColor=colors.HexColor("#0D47A1"))
#     section_style = ParagraphStyle('section', parent=styles['Heading2'], fontSize=11, textColor=colors.HexColor("#0D47A1"))
#     body_style = ParagraphStyle('body', parent=styles['BodyText'], fontSize=10)

#     story = []
#     story.append(Paragraph("AI Diagnostic Center", title_style))
#     story.append(Paragraph("MRI Brain Tumor Analysis Report", section_style))
#     story.append(Spacer(1, 8))

#     story.append(Paragraph(f"<b>Patient Name:</b> {patient.get('name','')}", body_style))
#     story.append(Paragraph(f"<b>Age:</b> {patient.get('age','')}  &nbsp;&nbsp; <b>Gender:</b> {patient.get('gender','')}", body_style))
#     story.append(Spacer(1, 6))

#     story.append(Paragraph("<b>Findings</b>", section_style))
#     # Preserve newlines
#     for chunk in report_text.split("\n\n"):
#         story.append(Paragraph(chunk.replace("\n", "<br/>"), body_style))
#         story.append(Spacer(1, 4))

#     story.append(Spacer(1, 8))

#     # Images row
#     imgs = []
#     try:
#         imgs.append(RLImage(original_img, width=240, height=240))
#     except Exception:
#         imgs.append(Paragraph("Original image not available", body_style))
#     try:
#         if mask_img and os.path.exists(mask_img):
#             imgs.append(RLImage(mask_img, width=240, height=240))
#         else:
#             imgs.append(Paragraph("No segmentation mask", body_style))
#     except Exception:
#         imgs.append(Paragraph("Mask not available", body_style))

#     story.append(Table([imgs], hAlign='CENTER'))
#     story.append(Spacer(1, 8))

#     story.append(Paragraph(f"<font size=8>This is an AI-assisted radiology evaluation. Generated on: {datetime.now().strftime('%d %b %Y %I:%M %p')}</font>", body_style))

#     doc.build(story)
#     return output_path

# # -------------------------
# # STREAMLIT UI
# # -------------------------
# st.set_page_config(page_title="AI MRI Diagnostic", layout="wide", initial_sidebar_state="expanded")

# # session_state defaults
# if "last_analysis" not in st.session_state:
#     st.session_state.last_analysis = {}
# if "last_report" not in st.session_state:
#     st.session_state.last_report = ""
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Sidebar navigation
# with st.sidebar:
#     if os.path.exists(LOGO_PATH):
#         st.image(LOGO_PATH, width=180)
#     st.title("Navigation")
#     page = st.radio("Navigation Menu",["Home", "Analysis", "Assistant", "About"],label_visibility="hidden")


# # --------- HOME PAGE ----------
# if page == "Home":
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         if os.path.exists(LOGO_PATH):
#             st.image(LOGO_PATH, width=220)
#         else:
#             st.markdown("<h2 style='color:#0D47A1'>AI MRI</h2>", unsafe_allow_html=True)
#     with col2:
#         st.markdown("<h1 style='color:#0D47A1'>MRI Brain Tumor Analysis</h1>", unsafe_allow_html=True)
#         st.write(
#             "This platform combines a CNN-based classifier, a U-Net segmentation model, and an LLM-assisted radiology assistant (Gemini) to produce professional MRI reports and downloadable PDFs."
#         )
#         st.markdown("**Quick Steps:**\n\n1. Go to *Analysis* → Upload MRI → Run Analysis.\n2. Generate & Download PDF report.\n3. Use *Assistant* to ask follow-up questions about the report (session memory).")
#     st.write("")
#     st.info("For a major project review: keep your models in the configured `models/` folder and place outputs in the `downloads/` folder. Edit paths at the top of this script if needed.")

# # --------- ANALYSIS PAGE ----------
# elif page == "Analysis":
#     st.header("MRI Analysis")
#     st.write("Upload a single MRI image (jpg/png). The app will run detection and segmentation (if segmentation model present).")
#     uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

#     # optional: sample/test image file path shown for convenience
#     st.markdown(f"**Models:** Detection = {'Loaded' if detection_model else 'Not loaded'}, Segmentation = {'Loaded' if segmentation_model else 'Not loaded'}")

#     if uploaded:
#         # save to temp file
#         temp_path = os.path.join(OUTPUT_DIR, f"temp_upload{os.path.splitext(uploaded.name)[1]}")
#         with open(temp_path, "wb") as f:
#             f.write(uploaded.getbuffer())

#         st.image(temp_path, caption="Uploaded MRI", use_column_width=True)

#         if st.button("Run Analysis"):
#             with st.spinner("Running detection & segmentation..."):
#                 try:
#                     result = analyze_mri_local(temp_path)
#                     st.session_state.last_analysis = result
#                     st.success("Analysis completed")
#                 except Exception as e:
#                     st.error(f"Analysis failed: {e}")
#                     result = None

#             if result:
#                 st.subheader("Results")
#                 st.markdown(f"- **Tumor Type:** `{result['predicted_label'].upper()}`")
#                 st.markdown(f"- **Confidence:** `{result['confidence']:.2f}%`")
#                 st.markdown(f"- **Tumor Coverage:** `{result['tumor_percentage']:.2f}%`")

#                 if result.get("mask_image_path") and os.path.exists(result["mask_image_path"]):
#                     st.image(result["mask_image_path"], caption="Segmentation Mask", use_column_width=False)

#                 # Report generation UI
#                 st.markdown("---")
#                 st.subheader("Generate Radiology Report (LLM-assisted)")
#                 pname = st.text_input("Patient Name", "John Doe")
#                 page_age = st.number_input("Age", min_value=0, max_value=120, value=45)
#                 pgender = st.selectbox("Gender", ["Male", "Female", "Other"])

#                 if st.button("Generate & Download PDF Report"):
#                     patient = {"name": pname, "age": page_age, "gender": pgender}
#                     # build LLM prompt
#                     prompt = (
#                         f"You are an expert radiology assistant. Generate a professional MRI brain tumor report.\n\n"
#                         f"Patient: {pname}\nAge: {page_age}\nGender: {pgender}\n\n"
#                         f"Automated results:\n- Tumor: {result['predicted_label']}\n- Confidence: {result['confidence']:.2f}%\n- Coverage: {result['tumor_percentage']:.2f}%\n\n"
#                         "Provide Findings, Impression, Recommendations in clean medical language."
#                     )
#                     with st.spinner("Generating report (LLM)..."):
#                         report_text = gemini_generate(prompt)
#                     st.session_state.last_report = report_text
#                     st.success("Report generated")

#                     # Display findings on screen
#                     st.subheader("Findings / Impression / Recommendations")
#                     st.write(report_text)

#                     # Create PDF
#                     pdf_fname = os.path.join(OUTPUT_DIR, f"mri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
#                     try:
#                         pdf_path = create_pdf(report_text, result["image_path"], result.get("mask_image_path"), patient, output_path=pdf_fname)
#                         # Read PDF bytes and provide download button
#                         with open(pdf_path, "rb") as f:
#                             pdf_bytes = f.read()
#                         st.download_button("⬇ Download PDF Report", data=pdf_bytes, file_name=os.path.basename(pdf_path), mime="application/pdf")
#                     except Exception as e:
#                         st.error(f"PDF generation failed: {e}")

# # --------- ASSISTANT PAGE ----------
# elif page == "Assistant":
#     st.header("AI Assistant (ask about last report)")
#     if not st.session_state.last_report:
#         st.info("No report available. Generate a report on the Analysis page first.")
#     else:
#         st.markdown("Assistant has access to the most recent generated report (session memory). Ask short, clinically cautious questions.")
#         query = st.text_input("Your question", "")
#         if st.button("Ask") and query.strip():
#             # compose context
#             context = st.session_state.last_report
#             history = "\n".join([f"User: {q}\nAI: {a}" for q, a in st.session_state.chat_history])
#             prompt = (
#                 f"Radiology report:\n{context}\n\n"
#                 f"Conversation history:\n{history}\n\n"
#                 f"User question: {query}\n\n"
#                 "Answer clearly and safely in plain language. If uncertain, advise clinical correlation."
#             )
#             with st.spinner("Assistant is thinking..."):
#                 answer = gemini_generate(prompt)
#             st.session_state.chat_history.append((query, answer))
#         # render chat history newest-first
#         for q, a in reversed(st.session_state.chat_history):
#             st.markdown(f"**You:** {q}")
#             st.markdown(f"**Assistant:** {a}")

#         if st.button("Clear Conversation"):
#             st.session_state.chat_history = []

# # --------- ABOUT PAGE ----------
# elif page == "About":
#     st.title("About this Project")
#     st.markdown(
#         """
#         **AI MRI Brain Tumor System**  
#         - MRI Tumor Classification (CNN)  
#         - Tumor Segmentation (U-Net)  
#         - Gemini LLM-powered report generation & assistant (optional)  
#         - Apollo-style PDF report with images & findings  
#         - Designed for major project presentations and paper-prep
#         """
#     )
#     st.markdown(f"**Output folder:** `{OUTPUT_DIR}`")
#     st.markdown("**Notes & tips:**")
#     st.markdown(
#         "- Put your models at the paths defined at top of this script (DETECTION_MODEL_PATH, SEGMENTATION_MODEL_PATH).\n"
#         "- If Gemini key is not provided or config fails, the app will generate conservative fallback text instead of using LLM.\n"
#         "- Use VS Code terminal: `streamlit run app.py` to start locally. For Colab use the earlier LocalTunnel/Cloudflare cell workflow."
#     )
