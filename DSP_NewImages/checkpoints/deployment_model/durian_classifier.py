
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

class DurianRipenessClassifier:
    def __init__(self, model_path, class_info_path):
        """Initialize the classifier"""
        self.model = load_model(model_path)
        with open(class_info_path, 'r') as f:
            self.class_info = json.load(f)
        self.input_shape = self.class_info['input_shape']
        self.class_names = self.class_info['class_names']
        
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize and normalize
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path):
        """Predict ripeness of durian image"""
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {
                self.class_names[i]: float(pred) for i, pred in enumerate(predictions[0])
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = DurianRipenessClassifier(
        model_path=os.path.join(os.path.dirname(__file__), "durian_cnn_model.h5"),
        class_info_path=os.path.join(os.path.dirname(__file__), "class_info.json")
    )
    
    # Example prediction
    try:
        result = classifier.predict("example_durian.jpg")
        print(f"Prediction: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("All probabilities:")
        for class_name, prob in result['all_predictions'].items():
            print(f"  {class_name}: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")
