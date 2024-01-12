import os
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
Bootstrap(app)

# Load your pre-trained model
labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}
model=load_model('saved_model.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img=img.resize((256,256))
    img = np.asarray(img, dtype= np.float32)
    img /= 255  # Normalize image array
    img = img.reshape(-1,256,256,3)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Make predictions using your pre-trained model
            prediction = model.predict(processed_image)
            prediction = np.argmax(prediction)
            # You might want to process the prediction further based on your model's output format
            predict=labels[prediction]
            # Remove the uploaded file after processing
            os.remove(file_path)

            # Add background color change based on prediction (example)
            return render_template('index.html',prediction=predict)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
