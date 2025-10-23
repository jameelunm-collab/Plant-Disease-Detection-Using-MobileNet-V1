import numpy as np
import tensorflow as tf
from PIL import Image

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# --- Get input/output details ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# --- Class labels ---
labels = [
    'Tomato Healthy',
    'Tomato Septoria Leaf Spot',
    'Tomato Bacterial Spot',
    'Tomato Blight',
    'Cabbage Healthy',
    'Tomato Spider Mite',
    'Tomato Leaf Mold',
    'Tomato Yellow Leaf Curl Virus',
    'Soy Frogeye Leaf Spot',
    'Soy Downy Mildew',
    'Maize Ravi Corn Rust',
    'Maize Healthy',
    'Maize Grey Leaf Spot',
    'Maize Lethal Necrosis',
    'Soy Healthy',
    'Cabbage Black Rot'
]

# --- Load & preprocess image ---
def predict(imag):
    img = Image.open(imag).resize((300, 300))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)  

    # --- Run inference ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()    

    # --- Get and dequantize output ---
    output_data = interpreter.get_tensor(output_details[0]['index'])
    scale, zero_point = output_details[0]['quantization']
    output_float = (output_data.astype(np.float32) - zero_point) * scale    

    # --- Get prediction ---
    predicted_class = int(np.argmax(output_float))
    confidence = float(np.max(output_float))    
    return {"Predicted Class Index": predicted_class,
            "Predicted Label": labels[predicted_class],
            "Confidence": confidence}
