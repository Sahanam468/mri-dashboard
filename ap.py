import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import cv2

# ---------------------------------------
# 1. MODEL PATHS  (update for your PC)
# ---------------------------------------
DETECTION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\best_inceptionv3_tumor.h5"
SEGMENTATION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\tumor_segmentation_unet.h5"

class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ---------------------------------------
# 2. LOAD MODELS
# ---------------------------------------
print("Loading models...")

try:
    detection_model = load_model(DETECTION_MODEL_PATH)
    print("✔ Detection model loaded")
except Exception as e:
    print("❌ Detection model error:", e)
    detection_model = None

try:
    segmentation_model = load_model(SEGMENTATION_MODEL_PATH, compile=False)
    print("✔ Segmentation model loaded")
except Exception as e:
    print("❌ Segmentation model error:", e)
    segmentation_model = None


# ---------------------------------------
# 3. FIXED ANALYSIS PIPELINE
# ---------------------------------------
def analyze_mri(img_path: str):
    print(f"\n🧠 Analyzing MRI image: {img_path}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # --- DETECTION PREPROCESSING ---
    img = image.load_img(img_path, target_size=(224, 224))
    arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    preds = detection_model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    label = class_labels[idx]
    confidence = float(preds[idx] * 100)

    print(f"🔍 Tumor Type: {label}  ({confidence:.2f}%)")

    tumor_pct = 0.0
    mask_path = None

    # ---------------------------------------
    #  SEGMENTATION (WITH FIXES)
    # ---------------------------------------
    if label != "no_tumor" and segmentation_model is not None:

        # Load grayscale and enhance contrast
        seg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        seg = cv2.equalizeHist(seg)  # improves segmentation
        seg = cv2.resize(seg, (128, 128))

        seg_arr = seg.astype("float32") / 255.0
        seg_arr = np.expand_dims(seg_arr, axis=-1)
        seg_arr = np.expand_dims(seg_arr, axis=0)

        mask_pred = segmentation_model.predict(seg_arr, verbose=0)[0]

        # remove channel dim
        if mask_pred.ndim == 3:
            mask = mask_pred[:, :, 0]
        else:
            mask = mask_pred

        # print statistics
        print("Mask min/max:", mask.min(), mask.max())

        # LOWER threshold (since your model outputs are low)
        binary = (mask > 0.15).astype(np.uint8)   # IMPORTANT FIX

        tumor_pct = float(binary.sum() / binary.size * 100)

        # save mask
        mask_path = img_path.replace(".jpg", "_mask.png")
        plt.imsave(mask_path, binary * 255, cmap="gray")

        print(f"📌 Tumor Coverage: {tumor_pct:.2f}%")
        print(f"🖼 Mask saved → {mask_path}")

    return {
        "image_path": img_path,
        "predicted_label": label,
        "confidence": confidence,
        "tumor_percentage": tumor_pct,
        "mask_image_path": mask_path
    }


# ---------------------------------------
# 4. TEST RUN (update your test image)
# ---------------------------------------
if __name__ == "__main__":
    test_image = r"C:\Users\RAKSHITHA SRINIVASA\Downloads\Te-gl_0010.jpg"
    result = analyze_mri(test_image)
    print("\nFinal Output:")
    print(result)



















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
DETECTION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\best_inceptionv3_tumor.h5"
SEGMENTATION_MODEL_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\models\tumor_segmentation_unet.h5"
LOGO_PATH = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\logo.png"
PDF_OUTPUT_DIR = r"C:\Users\RAKSHITHA SRINIVASA\Desktop\mri-dashboard\downloads"

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
      "<div style='text-align:center; margin-bottom:15px;'>"
      "<img src='https://i.gifer.com/HesE.gif' width='280'>"
    #   "<p style='font-size:15px; color:grey;'>AI-powered brain scan animation</p>"
      "</div>",
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
This platform performs:
- **MRI Tumor Detection** using InceptionV3.
- **Tumor Segmentation** using U-Net.
- **Radiology Report Generation** using Google Gemini.
- **PDF Report** with Apollo-style formatting.
""")
st.markdown("**Notes:**\n- This is a research/demo tool. Always confirm with clinical specialist.")

# End of app.py