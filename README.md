# 🌿 Plant Disease Detection using MobileNetV1

## 📘 Overview
This project develops a **lightweight deep learning model** using **MobileNetV1** to classify plant leaf diseases.  
The model identifies **16 categories** across multiple crops — **Tomato, Maize, Soy, and Cabbage** — distinguishing between healthy and diseased leaves efficiently.

---

## 🎯 Objective
The objective is to develop a lightweight deep learning model using **MobileNetV1** to accurately classify plant leaf images into 16 categories, including healthy and diseased leaves of tomato, maize, soy, and cabbage, enabling efficient and real-time detection of crop diseases for improved farm management.

---

## 🏷️ Class Labels
| Crop | Categories |
|------|-------------|
| **Tomato** | Healthy, Septoria Leaf Spot, Bacterial Spot, Blight, Spider Mite, Leaf Mold, Yellow Leaf Curl Virus |
| **Maize**  | Healthy, Ravi Corn Rust, Grey Leaf Spot, Lethal Necrosis |
| **Soy**    | Healthy, Frogeye Leaf Spot, Downy Mildew |
| **Cabbage**| Healthy, Black Rot |

---

## 🧠 Model Architecture
The model is based on **MobileNetV1**, a CNN optimized for mobile and embedded devices.  
**Key Features:**
- Depthwise Separable Convolutions  
- Reduced number of parameters  
- High accuracy with minimal computation  
- Real-time inference capability  

---

## 🗂️ Dataset
The dataset contains labeled images of healthy and diseased leaves from **Tomato, Maize, Soy, and Cabbage** crops.  
All images were preprocessed and resized to **224×224 pixels** before training.

---

## ⚙️ Technologies Used
- **Python**
- **TensorFlow / Keras**
- **NumPy**

---

## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/yourusername/plant-disease-detection.git

# Navigate to the project directory
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt

# Test the model
python test.py

# Run the web app
python app.py
