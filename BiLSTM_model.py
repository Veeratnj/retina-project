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

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
    y_test_classes = np.argmax(y_test, axis=1)  # Convert true labels to class labels
    report = classification_report(y_test_classes, y_pred_classes)
    print(report)

def evaluate_model(model, X_test, y_test):
    try:
      # Evaluate the model
      y_pred = model.predict(X_test)
      y_pred_classes = np.argmax(y_pred, axis=1)
      y_test_classes = np.argmax(y_test, axis=1)

      # Generate a classification report
      report = classification_report(y_test_classes, y_pred_classes)
      print(report)

      # Get class labels from y_test
      class_labels = [str(label) for label in np.unique(y_test_classes)]

      # Create a confusion matrix
      cm = confusion_matrix(y_test_classes, y_pred_classes)

      # Plot the confusion matrix
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.title('Confusion Matrix')
      plt.show()
    except Exception as e:
      print('matplotlib is not available for this platform')



def retinex_preprocess(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (64, 64))
    gray_image = (gray_image - gray_image.mean()) / gray_image.std()  # Normalize
    return gray_image

def data_preprocessing(dataset_dir, csv_file_path):
    images = []
    labels = []

    try:
        labels_df = pd.read_csv(csv_file_path)
        image_filenames = labels_df.iloc[:, 0]

        for index, image_filename in enumerate(image_filenames):
            image_path = os.path.join(dataset_dir, image_filename)

            if os.path.isfile(image_path):
                image = cv2.imread(image_path)
                gray_image = retinex_preprocess(image)
                images.append(gray_image)
                class_labels = labels_df.iloc[index, 1:].values.astype(np.float32)  # Ensure labels are float
                labels.append(class_labels)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels
    except Exception as e:
        print("Error reading CSV file:", e)

def model_training(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define and compile the model
    model = Sequential()
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Add fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    evaluate_model(model=model,X_test=X_test,y_test=y_test)
    # Save the trained model
    model.save('trained_BiLSTM.h5')

    return history

def detect_diabetes(input_image_path):
    # Load the trained BiLSTM-MSRCP model
    model = load_model('trained_BiLSTM.h5')

    # Preprocess the input image
    input_image = cv2.imread(input_image_path)
    gray_image = retinex_preprocess(input_image)

    # Reshape the image to match the model input shape
    gray_image = np.reshape(gray_image, (1, 64, 64, 1))

    # Pass the preprocessed image to the model to predict the class labels
    predicted_labels = model.predict(gray_image)

    # Return the predicted class labels
    return predicted_labels

# Example usage:
if __name__ == "__main__":
    # Define the paths to your dataset directory and CSV file
    dataset_dir = 'dataset'
    csv_file_path = 'dataset\_classes.csv'

    # Step 1: Data Preprocessing
    images, labels = data_preprocessing(dataset_dir, csv_file_path)

    # Step 4: Model Training
    history = model_training(images, labels)

    print("Training History:", history)

    # Step 5: Detect Diabetes (Example)
    input_image_path = 'hi.jpg'  # Replace with the path to your input image
    predicted_labels = detect_diabetes(input_image_path)

    print("your input img result:", predicted_labels)




def plot_class_distribution(csv_file_path):
    # Load the CSV file into a DataFrame
    labels_df = pd.read_csv(csv_file_path)
    
    # Get the class labels from the DataFrame
    class_labels = labels_df.iloc[:, 1:].columns
    
    # Count the number of images in each class
    class_counts = labels_df.iloc[:, 1:].sum().values
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Class Distribution')
    plt.show()

# Call the function
    plot_class_distribution('dataset\_classes.csv')