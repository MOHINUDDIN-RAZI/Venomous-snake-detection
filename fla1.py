from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = load_model('best_model.h5')

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(filepath):
    img = Image.open(filepath)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']

    # Check if the file is not empty
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file has an allowed extension
    if not allowed_file(uploaded_file.filename):
        return jsonify({'error': 'Invalid file extension. Only JPEG or PNG files are allowed.'})

    # Securely save the file
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(filepath)

    # Process the uploaded image
    img, img_array = load_and_preprocess_image(filepath)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = int(np.round(prediction[0][0]))  # Round to 0 or 1
    final_class = 'Venomous' if predicted_class == 1 else 'Non-Venomous'

    # Get probabilities for individual features
    eyes_percentage, head_percentage, tongue_percentage, pit_percentage, fang_percentage = prediction[1:]

    # Save the PIL image
    img.save('static/uploaded_image.jpg')

    return render_template('index.html', final_class=final_class, eyes_percentage=eyes_percentage,
                           head_percentage=head_percentage, tongue_percentage=tongue_percentage,
                           pit_percentage=pit_percentage, fang_percentage=fang_percentage, uploaded_image=True)

@app.route('/display_image')
def display_image():
    return send_file('static/uploaded_image.jpg', mimetype='image/jpg')

@app.route('/image1')
def image1():
    return send_file('templates\king-cobra.jpg', mimetype='image/jpg')

@app.route('/image2')
def image2():
    return send_file('templates\snakeinfo2.jpeg', mimetype='image/jpg')

@app.route('/image3')
def image3():
    return send_file('templates\Russells-viper-snake.jpg', mimetype='image/jpg')

@app.route('/image4')
def image4():
    return send_file('templates\Snake-bite.jpg', mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
