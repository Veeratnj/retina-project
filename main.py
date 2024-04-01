import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns

from BiLSTM_model import retinex_preprocess

def detect_diabetes(img,modelPath):

    # Load the trained BiLSTM-MSRCP model
    model = load_model(modelPath)

    # Preprocess the input image
    input_image = cv2.imread(img)
    gray_image = retinex_preprocess(input_image)

    # Reshape the image to match the model input shape
    gray_image = np.reshape(gray_image, (1, 64, 64, 1))

    # Pass the preprocessed image to the model to predict the class labels
    predicted_labels = model.predict(gray_image)

    # Return the predicted class labels
    return predicted_labels

img='hi.jpg'
modelPath=r'trained_BiLSTM.h5'
output = detect_diabetes(img,modelPath)
print(output)
max_value = np.max(output)
max_index = np.argmax(output)
print(max_value,max_index)
label = ['Mild','Moderate','No_DR','Proliferate_DR','Severe']
print(label[max_index])
o=cv2.imread(img)
cv2.imshow(f"{label[max_index]}",o)
cv2.waitKey(0)





