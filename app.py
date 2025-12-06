import streamlit as st
import os
import pickle
import json
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Durian Ripeness Classifier",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR = "output"
MODEL_PKL_PATH = os.path.join(OUTPUT_DIR, "durian_model.pkl")

# CLASS_INFO_PATH = r"C:\Users\torre\Desktop\FinalPIT_Electives\electives1-finalpit\DSP_NewImages\checkpoints\deployment_model\class_info.json"
CLASS_INFO_PATH = r"DSP_NewImages\checkpoints\deployment_model\class_info.json"

# ✅ SAMPLE IMAGE USED FOR AUTO TEST
SAMPLE_IMAGE_PATH = r"ElectivesSampleImages\Electives1_Sampledata.jpg"

# ✅ SET EXPECTED CLASS HERE
EXPECTED_CLASS = "Ripe"   # Change to: "Unripe" or "Overripe"


# =============================================================================
# FIRST RUN FLAG
# =============================================================================
if "first_run" not in st.session_state:
    st.session_state.first_run = True


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def figure_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close(fig)
    return pil_img

def create_color_analysis_visualization(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))
    ax1.imshow(rgb_image); ax1.set_title('RGB Analysis'); ax1.axis('off')
    ax2.imshow(hsv[:,:,0], cmap='hsv'); ax2.set_title('Hue Distribution'); ax2.axis('off')
    ax3.imshow(hsv[:,:,1], cmap='viridis'); ax3.set_title('Saturation Level'); ax3.axis('off')
    ax4.imshow(lab[:,:,0], cmap='gray'); ax4.set_title('Lightness Pattern'); ax4.axis('off')
    plt.tight_layout()
    return figure_to_pil(fig)

def create_texture_analysis_visualization(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))
    ax1.imshow(rgb_image); ax1.set_title('Original'); ax1.axis('off')
    ax2.imshow(edges, cmap='gray'); ax2.set_title('Edge Detection'); ax2.axis('off')
    ax3.imshow(np.abs(sobelx), cmap='hot'); ax3.set_title('Horizontal Texture'); ax3.axis('off')
    ax4.imshow(np.abs(sobely), cmap='cool'); ax4.set_title('Vertical Texture'); ax4.axis('off')
    plt.tight_layout()
    return figure_to_pil(fig)

def create_probability_visualization(predictions, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    bars = ax.bar(class_names, predictions, color=colors, alpha=0.8)
    ax.set_ylabel('Probability')
    ax.set_title('Class Probability Distribution')
    ax.set_ylim(0, 1)
    for bar, prob in zip(bars, predictions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    return figure_to_pil(fig)

def generate_reasoning(confidence, predicted_class):
    reasoning_map = {
        "Overripe": "Dark brown/yellow hues, soft texture, and less prominent spines detected.",
        "Ripe": "Balanced color, moderate spine definition, and ideal texture patterns detected.",
        "Unripe": "Bright green tone, sharp spines, and firm texture detected."
    }

    if confidence > 0.9:
        confidence_text = "🔴 VERY HIGH CONFIDENCE"
    elif confidence > 0.7:
        confidence_text = "🟡 HIGH CONFIDENCE"
    elif confidence > 0.5:
        confidence_text = "🟢 MODERATE CONFIDENCE"
    else:
        confidence_text = "⚪ LOW CONFIDENCE"

    return f"{reasoning_map.get(predicted_class, '')}\n\n**{confidence_text}**"


# =============================================================================
# CORE PREDICTION
# =============================================================================

def predict_durian_ripeness_detailed(model, class_info, pil_image):

    process_images = {}

    open_cv_image = np.array(pil_image.convert('RGB'))
    original_image = open_cv_image[:, :, ::-1].copy()

    resized_image = cv2.resize(original_image, (224, 224))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image.astype('float32') / 255.0

    process_images["🎨 Color Analysis"] = create_color_analysis_visualization(rgb_image)
    process_images["🔍 Texture Analysis"] = create_texture_analysis_visualization(rgb_image)

    batch_image = np.expand_dims(normalized_image, axis=0)
    predictions = model.predict(batch_image, verbose=0)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_info['class_names'][predicted_class_index]
    confidence = float(np.max(predictions))

    process_images["🤖 Prediction"] = create_probability_visualization(
        predictions, class_info['class_names']
    )

    return {
        'predicted_class': predicted_class_name,
        'confidence': confidence,
        'process_images': process_images,
        'reasoning': generate_reasoning(confidence, predicted_class_name),
        'success': True
    }


# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PKL_PATH) or not os.path.exists(CLASS_INFO_PATH):
        return None, None
    with open(MODEL_PKL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(CLASS_INFO_PATH, 'r') as f:
        class_info = json.load(f)
    return model, class_info

model, class_info = load_resources()


# =============================================================================
# UI
# =============================================================================

st.title("🍈 Durian Ripeness Classifier")
st.markdown("AI-powered tool to analyze durian ripeness from an image.")


if not model or not class_info:
    st.error("❌ Model or class information not found.")

else:

    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a durian image...",
            type=["jpg", "jpeg", "png"]
        )

        if st.session_state.first_run:
            st.success("✅ Sample image auto-loaded on first run.")

    # =================== AUTO LOAD SAMPLE =====================

    if uploaded_file is None and st.session_state.first_run:
        if os.path.exists(SAMPLE_IMAGE_PATH):
            pil_image = Image.open(SAMPLE_IMAGE_PATH)
            sample_mode = True
        else:
            st.error("❌ Sample image not found.")
            pil_image = None
            sample_mode = False
    else:
        pil_image = Image.open(uploaded_file) if uploaded_file else None
        sample_mode = False


    if pil_image is not None:

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.image(
                pil_image,
                caption="Current Image",
                use_container_width=True
            )

        with col2:
            if st.button("🔍 Analyze Ripeness") or (sample_mode and st.session_state.first_run):

                with st.spinner("Analyzing..."):
                    result = predict_durian_ripeness_detailed(
                        model,
                        class_info,
                        pil_image
                    )

                st.subheader("🏆 RESULT")
                st.markdown(f"### **{result['predicted_class']}**")
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                st.info(result['reasoning'])

                for key, img in result['process_images'].items():
                    st.subheader(key)
                    st.image(img, use_container_width=True)


                # TURN OFF AUTO-RUN AFTER FIRST SAMPLE
                st.session_state.first_run = False

    else:
        st.info("Please upload an image to begin.")
