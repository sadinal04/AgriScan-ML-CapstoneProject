# AgriScan Capstone Project

**AgriScan** adalah aplikasi berbasis **Streamlit** untuk mendeteksi penyakit pada berbagai tanaman menggunakan model machine learning berbasis citra daun. Model dilatih dengan **transfer learning ResNet50** dan **tuning hyperparameter manual** untuk meningkatkan performa klasifikasi.

🌱 **Aplikasi sudah online, bisa diakses di:**  
[👉 AgriScan Streamlit App](https://agriscan-capstoneproject.streamlit.app/)

---

## 🌾 Tanaman yang Didukung

Model mendeteksi penyakit pada daun dari tanaman berikut:
- 🌾 **Padi**
- 🍎 **Apel**
- 🌽 **Jagung**
- 🍇 **Anggur**
- 🥔 **Kentang**
- 🍅 **Tomat**

---

## 📂 Struktur Repository

| Nama | Deskripsi |
|-------|-----------|
| `labels/` | Label atau metadata untuk klasifikasi penyakit tanaman. |
| `tfjs_model_PlantLeaf99/` | Model PlantLeaf99 dalam format TensorFlow.js untuk inferensi berbasis web. |
| `tfjs_model_plant_vs_nonplant/` | Model TensorFlow.js untuk membedakan gambar tanaman vs bukan tanaman. |
| `tfjs_model_riceLeaf/` | Model TensorFlow.js untuk mendeteksi penyakit daun padi. |
| `.gitattributes` | Konfigurasi atribut Git (misalnya untuk LFS). |
| `.gitignore` | File yang menentukan file/folder yang diabaikan Git. |
| `app.py` | Aplikasi utama Streamlit untuk antarmuka pengguna dan prediksi. |
| `plantvillagenotebook.ipynb` | Notebook Jupyter untuk pelatihan model menggunakan **ResNet50 transfer learning** dan tuning manual. |
| `requirements.txt` | Daftar library Python yang diperlukan untuk menjalankan aplikasi. |

---

## 🔬 Model Machine Learning

- **Arsitektur:** ResNet50 (transfer learning)
- **Tuning:** Hyperparameter dioptimasi secara manual.
- **Output model:** `.keras` (untuk aplikasi) dan TensorFlow.js (untuk deployment web).
- **Dataset:** Citra daun dari tanaman Padi, Apel, Jagung, Anggur, Kentang, Tomat.

---

## 🚀 Cara Menjalankan Secara Lokal


```bash
## clone repo
git clone https://github.com/username/repo.git
cd repo

## Install dependensi:
pip install -r requirements.txt

##Jalankan Streamlit:
streamlit run app.py
```

## 🔗 Link Aplikasi

👉 [AgriScan Streamlit App](https://agriscan-capstoneproject.streamlit.app/)

---

## 📝 Catatan

- Model mendukung klasifikasi penyakit pada 6 jenis tanaman: **Padi**, **Apel**, **Jagung**, **Anggur**, **Kentang**, dan **Tomat**.
- Aplikasi sudah siap digunakan untuk deteksi berbasis gambar daun melalui antarmuka sederhana di **Streamlit**.
- Model juga tersedia dalam **TensorFlow.js** untuk deployment berbasis browser.
