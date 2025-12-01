
import os
import pickle
from tensorflow.keras.models import load_model

# --- Configuration ---
DEPLOYMENT_DIR = r"C:\DSP_MachineLearning\DSP_NewImages\checkpoints\deployment_model"
MODEL_H5_PATH = os.path.join(DEPLOYMENT_DIR, "durian_cnn_model.h5")
OUTPUT_DIR = r"C:\DSP_MachineLearning\output"
MODEL_PKL_PATH = os.path.join(OUTPUT_DIR, "durian_model.pkl")

# --- Main script ---
if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_H5_PATH):
        print(f"❌ Error: Model file not found at {MODEL_H5_PATH}")
    else:
        try:
            # 1. Load the Keras model from the .h5 file
            print(f"🧠 Loading Keras model from: {MODEL_H5_PATH}")
            model = load_model(MODEL_H5_PATH)
            print("✅ Model loaded successfully.")

            # 2. Use pickle to dump the model object to a .pkl file
            print(f"📦 Pickling model to: {MODEL_PKL_PATH}")
            with open(MODEL_PKL_PATH, 'wb') as f:
                pickle.dump(model, f)
            print("✅ Model successfully saved as a .pkl file.")
            print("\n---")
            print("You can now load this model in another script using:")
            print("with open('{}', 'rb') as f:".format(MODEL_PKL_PATH.replace('\\', '/')))
            print("    loaded_model = pickle.load(f)")
            print("---")

        except Exception as e:
            print(f"❌ An error occurred: {e}")
