import os
import numpy as np
import pandas as pd
from datetime import date
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# Define paths
train_data_path = "Gem Dataset/train"
test_data_path = "Gem Dataset/test"
model_path = 'gem_classification_model.keras'

# Function to train and save the model
def train_model():
    # Create ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    # Load training and validation data
    train_gen = datagen.flow_from_directory(train_data_path, 
                                            target_size=(150, 150), 
                                            batch_size=32, 
                                            class_mode='categorical',
                                            subset='training')

    val_gen = datagen.flow_from_directory(train_data_path, 
                                          target_size=(150, 150), 
                                          batch_size=32, 
                                          class_mode='categorical',
                                          subset='validation')
    
    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_gen, validation_data=val_gen, epochs=30)

    # Save the model
    model.save(model_path)
    
    return model, history, train_gen

# Load or train the model
if os.path.exists(model_path):
    model = load_model(model_path)
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_gen = datagen.flow_from_directory(train_data_path, 
                                            target_size=(150, 150), 
                                            batch_size=32, 
                                            class_mode='categorical')
    history = None  # Placeholder, won't be used if the model is already trained
else:
    model, history, train_gen = train_model()

class_names = list(train_gen.class_indices.keys())

# Function to predict the class of an image
def predict(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    prediction = model.predict(img)
    return class_names[np.argmax(prediction)], prediction

# Function to evaluate model performance
def evaluate_model():
    test_gen = datagen.flow_from_directory(test_data_path, 
                                           target_size=(150, 150), 
                                           batch_size=32, 
                                           class_mode='categorical',
                                           shuffle=False)

    loss, accuracy = model.evaluate(test_gen)
    return loss, accuracy, test_gen

# Function to plot confusion matrix
def plot_confusion_matrix(test_gen):
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    st.pyplot(plt)

# Streamlit Interface
st.set_page_config(page_title="Gem AI", layout="wide")

# Custom CSS for sidebar and font colors
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #2E3B4E;
        color: white;
    }
    .stTextInput label, .stButton button, .stFileUploader label, .stFileUploader div div {
        color: white;
    }
    .stButton button:hover {
        background-color: #4CAF50; /* Green */
        color: white;
    }
    .stAlert div[role="alert"] {
        background-color: #4CAF50;
        color: white;
    }
    .stApp {
        color: gray;
    }
    .stMarkdown h1 {
        font-size: 2.5rem;
        font-family: 'Gem', sans-serif;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Gem AI')
st.write("Upload an image of a gem, and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = load_img(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
    with col2:
        st.write("Classifying...")
        with st.spinner('Classifying...'):
            time.sleep(2)
            label, _ = predict(uploaded_file)
            st.success(f'Prediction: {label}')
            st.balloons()

            # Show an example image from the dataset
            example_image_path = os.path.join(train_data_path, label, os.listdir(os.path.join(train_data_path, label))[0])
            example_image = load_img(example_image_path)
            st.image(example_image, caption=f'Example of {label}', use_column_width=False, width=100)

    # Additional Code to Display Model Performance and Data Visualizations
    st.header("Model Performance and Data Visualization")

    # Evaluate model and plot confusion matrix
    st.subheader("Confusion Matrix")
    _, _, test_gen = evaluate_model()
    plot_confusion_matrix(test_gen)

    # Plotting training history graph
    if history:
        st.subheader("Training History")
        st.subheader("Accuracy and Loss over Epochs")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')

        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')

        st.pyplot(fig)

    # Display class distribution in the training set
    train_counts = pd.Series(train_gen.classes).value_counts().sort_index()
    train_class_counts = pd.DataFrame({
        'Class': [class_names[i] for i in train_counts.index],
        'Count': train_counts.values
    })

    st.subheader("Class Distribution in Training Data")
    chart = alt.Chart(train_class_counts).mark_bar().encode(
        x=alt.X('Class', sort=None),
        y='Count',
        color='Class'
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(chart)

    # Display class distribution in the test set
    test_counts = pd.Series(test_gen.classes).value_counts().sort_index()
    test_class_counts = pd.DataFrame({
        'Class': [class_names[i] for i in test_counts.index],
        'Count': test_counts.values
    })

    st.subheader("Class Distribution in Test Data")
    chart = alt.Chart(test_class_counts).mark_bar().encode(
        x=alt.X('Class', sort=None),
        y='Count',
        color='Class'
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(chart)

# Sidebar for additional options
st.sidebar.title("Gem AI")
st.sidebar.write("Date: " + str(date.today()))
st.sidebar.header("About Us")
st.sidebar.write("Gem AI is a state-of-the-art AI model for classifying various gemstones. Upload an image to see the magic!")
st.sidebar.header("Contact Us")
st.sidebar.write("Email: support@gemai.com")

# Streamlit sidebar interface to display model performance
with st.sidebar:
    st.header("Model Performance")
    loss, accuracy, _ = evaluate_model()

    st.metric(label="Test Accuracy", value=f"{accuracy:.2%}")
    st.metric(label="Test Loss", value=f"{loss:.4f}")

# Function to plot training history
def plot_history(history):
    if history:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        st.pyplot(fig)

# Display training performance in sidebar
if history:
    st.sidebar.header('Training Performance')
    plot_history(history)