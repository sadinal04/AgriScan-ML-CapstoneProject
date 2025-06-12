import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import time

# Caching agar model tidak diload berulang
@st.cache_resource
def load_model_keras(model_path):
    start = time.time()
    model = load_model(model_path)
    end = time.time()
    st.success(f"Model loaded in {end - start:.2f} seconds")
    return model

# Fungsi untuk load label dari file txt
def load_labels(label_file):
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# UI
st.title("🌿 Plant Disease Classification")
st.write("Upload an image of a leaf to predict the plant disease.")

# Pilihan jenis tanaman
plant_type = st.selectbox("Select Plant Type:", ["Padi", "Apel, Jagung, Anggur, Kentang, Tomat"])

# Load model dan label map sesuai pilihan
if plant_type == "Padi":
    model = load_model_keras("riceLeafModel.keras")
    label_map = load_labels("labels/rice_labels.txt")
else:
    model = load_model_keras("modelPlantLeaf99.keras")
    label_map = load_labels("labels/plantVillage_labels.txt")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    st.write("Classifying...")
    img = img.resize((224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predict
    prediction = model.predict(x)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    label = label_map[class_index]

    # Output utama
    st.markdown(f"### 🩺 Prediction: `{label}`")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    # Output confidence semua kelas
    st.subheader("🔍 All Class Confidences")
    confidences = prediction[0]
    confidence_dict = {label_map[i]: float(confidences[i]) for i in range(len(label_map))}
    sorted_confidences = dict(sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True))

    st.table(
        [{"Label": key, "Confidence (%)": f"{value * 100:.2f}"} for key, value in sorted_confidences.items()]
    )
