import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import time

# Caching model agar tidak diload berulang
@st.cache_resource
def load_model_keras(path):
    start = time.time()
    model = load_model(path)
    end = time.time()
    st.success(f"Model loaded from {path} in {end - start:.2f} seconds")
    return model

# Fungsi load label
def load_labels(label_file):
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Judul Aplikasi
st.title("🌿 Plant Disease Classification")
st.write("Upload an image of a leaf to predict the plant disease.")

# Pilihan tanaman
plant_type = st.selectbox("Select Plant Type:", ["Padi", "Apel, Jagung, Anggur, Kentang, Tomat"])

# Load model klasifikasi penyakit berdasarkan jenis tanaman
if plant_type == "Padi":
    disease_model = load_model_keras("riceLeafModel.keras")
    label_map = load_labels("labels/rice_labels.txt")
else:
    disease_model = load_model_keras("modelPlantLeaf99.keras")
    label_map = load_labels("labels/plantVillage_labels.txt")

# Load model deteksi tanaman atau bukan
binary_model = load_model_keras("plant_vs_nonplant.keras")
binary_labels = ["non-plant", "plant"]

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='📸 Uploaded Image', use_column_width=True)

    # Langkah 1: Cek apakah gambar tanaman
    st.write("🔍 Checking if the image contains a plant...")

    check_img = img.resize((224, 224))
    check_x = keras_image.img_to_array(check_img)
    check_x = np.expand_dims(check_x, axis=0)
    check_x = check_x / 255.0

    check_pred = binary_model.predict(check_x)
    check_class = np.argmax(check_pred)
    check_label = binary_labels[check_class]
    check_conf = np.max(check_pred)

    if check_label == "non-plant":
        st.error(f"🚫 The uploaded image is **not a plant**. Please upload a leaf image.")
    else:
        st.success(f"✅ Image classified as plant ({check_conf * 100:.2f}%)")
        st.write("🩺 Classifying plant disease...")

        # Langkah 2: Klasifikasi penyakit
        img_resized = img.resize((224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        prediction = disease_model.predict(x)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        label = label_map[class_index]

        st.markdown(f"### 🩺 Prediction: `{label}`")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        # Tampilkan seluruh confidence
        st.subheader("🔍 All Class Confidences")
        confidences = prediction[0]
        confidence_dict = {label_map[i]: float(confidences[i]) for i in range(len(label_map))}
        sorted_confidences = dict(sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True))

        st.table(
            [{"Label": key, "Confidence (%)": f"{value * 100:.2f}"} for key, value in sorted_confidences.items()]
        )
