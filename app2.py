import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from BiLSTM_model import retinex_preprocess

def detect_diabetes(img, model_path):
    # Load the selected model
    model = load_model(model_path)

    # Preprocess the input image
    gray_image = retinex_preprocess(img)

    # Reshape the image to match the model input shape
    gray_image = np.reshape(gray_image, (1, 64, 64, 1))

    # Pass the preprocessed image to the model to predict the class labels
    predicted_labels = model.predict(gray_image)

    # Return the predicted class labels
    return predicted_labels

def main():
    st.title('Diabetes Retinopathy Detection')
    st.write('Upload an image for prediction')

    # Dropdown for choosing the model
    selected_model = st.selectbox("Select Model", ["Model 1", "Model 2"])

    model_paths = {
        "Model 1": "trained_BiLSTM.h5",
        "Model 2": "trained_BiLSTM.h5",
    }

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image with original color
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Uploaded Image', width=200, channels='BGR')

        # Run prediction
        output = detect_diabetes(img, model_paths[selected_model])
        label = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
        max_index = np.argmax(output)
        prediction = label[max_index]

        with col2:
            st.image(img, width=200, channels='BGR')
            st.write('Prediction:', prediction)

if __name__ == "__main__":
    main()
