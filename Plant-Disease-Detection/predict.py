import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/plant_disease_cnn.keras"
IMAGE_SIZE = (128, 128)

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

model = tf.keras.models.load_model(MODEL_PATH)

image = Image.open(image_path).convert("RGB")
image = image.resize(IMAGE_SIZE)

image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array, verbose=0)

predicted_index = np.argmax(prediction[0])
confidence = prediction[0][predicted_index] * 100

print("\nPlant Disease Detection")
print("-----------------------")
print("Predicted Class:", class_names[predicted_index])
print("Confidence:", round(confidence, 2), "%")