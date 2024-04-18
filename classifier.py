import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle as pk
import pandas as pd
# Load best model
checkpoint_dir = 'ckpt_1_CNN_with_TF_VGG19_with_DataAug'
best_model = load_model('E://best_model//Animals_CNN_TF_VGG19_epoch_30_ES.h5')

# Load the categories
df = pd.read_csv('E://best_model//class_assignment_df_Animals_CNN_TF_VGG19_epoch_30_ES.csv')
df = df.sort_values(by='Allocated Number', ascending=True)
CATEGORIES = df['Category'].to_list()

# Load the used image height and width
with open('E://best_model//ckpt_1_CNN_with_TF_VGG19_with_DataAug_img_height.pkl', 'rb') as f:
    img_height_reload = pk.load(f)

with open('E://best_model//ckpt_1_CNN_with_TF_VGG19_with_DataAug_img_width.pkl', 'rb') as f:
    img_width_reload = pk.load(f)

# Function to predict image class
def predict_image_class(image):
    # Resize image to match model input size
    img_pred = cv2.resize(image, (img_height_reload, img_width_reload))
    img_pred = np.reshape(img_pred, [1, img_height_reload, img_width_reload, 3])

    # Predict class
    classes = np.argmax(best_model.predict(img_pred), axis=-1)
    predicted_class = CATEGORIES[int(classes[0])]
    return predicted_class

# Main Streamlit app
st.title('Manga vs Classic Art style Image Classifier')

uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict the class
    predicted_class = predict_image_class(image)
    st.write('Predicted Class:', predicted_class)
