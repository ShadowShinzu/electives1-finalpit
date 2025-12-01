
import streamlit as st
import os
import pickle
import json
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

# ===========================================================F==================
# PATH AND APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Durian Ripeness Classifier",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

OUTPUT_DIR = "output"
MODEL_PKL_PATH = os.path.join(OUTPUT_DIR, "durian_model.pkl")
CLASS_INFO_PATH = r"C:\DSP_MachineLearning\DSP_NewImages\checkpoints\deployment_model\class_info.json"


# =============================================================================
# VISUALIZATION & ANALYSIS FUNCTIONS (Adapted from Fin3.py)
# =============================================================================

def figure_to_pil(fig):
    """Convert matplotlib figure to a PIL Image"""
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
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{prob:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    return figure_to_pil(fig)

def generate_reasoning(confidence, predicted_class):
    reasoning_map = {
        "Overripe": "The model focused on dark brown/yellow hues, low spine prominence, and a softer texture appearance.",
        "Ripe": "The model identified a balanced yellow-green color, moderate spine definition, and texture patterns associated with peak ripeness.",
        "Unripe": "The model detected bright green coloration, sharp and prominent spines, and a firm texture, which are all key indicators of an unripe durian."
    }
    
    if confidence > 0.9:
        confidence_text = "🔴 VERY HIGH CONFIDENCE - Clear and distinct visual indicators were present."
    elif confidence > 0.7:
        confidence_text = "🟡 HIGH CONFIDENCE - Strong matching patterns were detected with minimal ambiguity."
    elif confidence > 0.5:
        confidence_text = "🟢 MODERATE CONFIDENCE - Some mixed features were present but a clear classification was made."
    else:
        confidence_text = "⚪ LOW CONFIDENCE - Mixed or unclear visual cues were detected. Manual verification is recommended."

    base_reasoning = reasoning_map.get(predicted_class, "The model analyzed the image based on its trained data.")
    return f"{base_reasoning}\n\n**Confidence Assessment:** {confidence_text}"


# =============================================================================
# CORE PREDICTION LOGIC
# =============================================================================

def predict_durian_ripeness_detailed(model, class_info, pil_image):
    """
    Predict durian ripeness using a loaded model and a PIL image.
    """
    process_steps = []
    process_images = {}

    # Step 1: Load original image
    open_cv_image = np.array(pil_image.convert('RGB'))
    original_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR
    original_shape = original_image.shape
    step_1_title = "📸 Original Image"
    process_steps.append({"step": 1, "title": step_1_title, "description": f"Loaded image with dimensions: {original_shape[1]}x{original_shape[0]} pixels"})
    process_images[step_1_title] = pil_image

    # Step 2: Resize and Preprocess
    resized_image = cv2.resize(original_image, (224, 224))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    step_2_title = "🔄 Image Resizing & Formatting"
    process_steps.append({"step": 2, "title": step_2_title, "description": "Resized to 224x224 pixels and converted to RGB for the model."})
    process_images[step_2_title] = Image.fromarray(rgb_image)

    # Step 3: Normalization
    normalized_image = rgb_image.astype('float32') / 255.0
    step_3_title = "📊 Pixel Normalization"
    process_steps.append({"step": 3, "title": step_3_title, "description": "Normalized pixel values to the [0, 1] range for model stability."})
    process_images[step_3_title] = Image.fromarray((normalized_image * 255).astype(np.uint8))

    # Step 4: Color and Texture Analysis
    step_4_title = "🎨 Color Analysis"
    process_steps.append({"step": 4, "title": step_4_title, "description": "Analyzed color distribution (Hue, Saturation) to identify ripeness indicators."})
    process_images[step_4_title] = create_color_analysis_visualization(rgb_image)
    
    step_5_title = "🔍 Texture Analysis"
    process_steps.append({"step": 5, "title": step_5_title, "description": "Analyzed surface texture and spine patterns using edge detection."})
    process_images[step_5_title] = create_texture_analysis_visualization(rgb_image)

    # Step 5: Model Prediction
    batch_image = np.expand_dims(normalized_image, axis=0)
    predictions = model.predict(batch_image, verbose=0)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_info['class_names'][predicted_class_index]
    confidence = float(np.max(predictions))
    
    step_6_title = "🤖 Neural Network Prediction"
    process_steps.append({"step": 6, "title": step_6_title, "description": f"The model processed deep features and made a prediction."})
    process_images[step_6_title] = create_probability_visualization(predictions, class_info['class_names'])

    # Step 7: Final Result
    reasoning = generate_reasoning(confidence, predicted_class_name)
    step_7_title = "📈 Final Result"
    process_steps.append({"step": 7, "title": step_7_title, "description": f"Classified as **{predicted_class_name}** with **{confidence:.1%}** confidence."})
    
    return {
        'predicted_class': predicted_class_name,
        'confidence': confidence,
        'all_probabilities': {class_info['class_names'][i]: float(p) for i, p in enumerate(predictions)},
        'process_steps': process_steps,
        'process_images': process_images,
        'reasoning': reasoning,
        'success': True
    }


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("🍈 Durian Ripeness Classifier")
st.markdown("An AI-powered tool to analyze the ripeness of a durian from an image, showing the visual analysis process.")

# --- Model and Data Loading ---
@st.cache_resource
def load_resources():
    """Load the model and class info, caching them for performance."""
    if not os.path.exists(MODEL_PKL_PATH) or not os.path.exists(CLASS_INFO_PATH):
        return None, None
    with open(MODEL_PKL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(CLASS_INFO_PATH, 'r') as f:
        class_info = json.load(f)
    return model, class_info

model, class_info = load_resources()

if not model or not class_info:
    st.error("❌ **Error:** Model or class info files not found. Please ensure `output/durian_model.pkl` and the corresponding `class_info.json` are in their correct paths.")
else:
    # --- Sidebar for Upload and Info ---
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose a durian image...", type=["jpg", "jpeg", "png"])
        
        st.divider() 
        
        with st.expander("🧠 About the Model"):
            st.markdown("""
            This app uses a **MobileNetV2** model fine-tuned to classify durians into three categories: **Unripe, Ripe, and Overripe**.
            
            The analysis involves several steps:
            1.  **Image Preprocessing**: Resizing and normalizing the image.
            2.  **Feature Analysis**: Examining color and texture patterns.
            3.  **AI Prediction**: Using the deep learning model to make a classification.
            4.  **Confidence Scoring**: Providing a probability distribution for the classes.
            """)

    # --- Main Content Area ---
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            st.header("Original Image")
            st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        with col2:
            st.header("Analysis Control")
            if st.button("🔍 Analyze Ripeness", use_container_width=True, type="primary"):
                with st.spinner("🤖 Performing AI analysis... please wait."):
                    result = predict_durian_ripeness_detailed(model, class_info, pil_image)

                st.header("🏆 Result")
                color = "#27ae60" if result['confidence'] > 0.7 else "#f39c12" if result['confidence'] > 0.5 else "#e74c3c"
                st.markdown(f"### Prediction: <span style='color:{color};'>**{result['predicted_class']}**</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{result['confidence']:.1%}`")

                st.subheader("💡 AI Reasoning")
                st.info(result['reasoning'])
        
                # Display detailed analysis in an expander
                with st.expander("🔄 Show Full Visual Analysis Process", expanded=True):
                    st.subheader("Step-by-Step AI Analysis")
                    for step in result['process_steps']:
                        # Create a side-by-side layout for each step
                        col1, col2 = st.columns([0.5, 0.5])
                        with col1:
                            st.write(f"**{step['title']}**")
                            st.write(step['description'])
                        
                        with col2:
                            if step['title'] in result['process_images']:
                                st.image(result['process_images'][step['title']], use_container_width=True)
                        st.divider()

    else:
        st.info("Please upload an image using the sidebar to begin the analysis.")
