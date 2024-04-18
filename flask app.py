from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle as pk
import pandas as pd
from io import BytesIO

app = Flask(__name__)
app.template_folder = 'E://best_model//templates'
# Load the pre-trained CNN model
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


@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['image']
        
        # Convert the file to a file-like object
        file_bytes = file.read()
        img = load_img(BytesIO(file_bytes), target_size=(img_height_reload, img_width_reload))

        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_pred = tf.image.resize(img_array, (img_height_reload, img_width_reload))
        img_pred = np.reshape(img_pred, [1, img_height_reload, img_width_reload, 3])

        # Predict class
        classes = np.argmax(best_model.predict(img_pred), axis=-1)
        predicted_class = CATEGORIES[int(classes[0])]

        # Make the prediction
        return render_template('index.html', result=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)