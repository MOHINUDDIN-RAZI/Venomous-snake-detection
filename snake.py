from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load pre-trained ResNet50 model for snake detection
model_resnet = ResNet50(weights='imagenet')

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


    # Load pre-trained ResNet50 model for snake detection
    model_resnet = ResNet50(weights='imagenet')

    # Load pre-trained best model for venomous snake classification
    model_venomous = load_model('best_model.h5')

    # Function to detect snakes in an image using ResNet50
    def detect_snakes_resnet(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make prediction
        preds = model_resnet.predict(x)
        # Decode prediction
        predictions = decode_predictions(preds, top=5)[0]
        
        # Check if any of the top predictions contains a snake class
        for pred in predictions:
            if 'snake' in pred[1]:
                return True, pred[2]  # Return True if snake detected along with confidence score
        return False, 0  # Return False if no snake detected

    # Function to classify if a detected snake is venomous
    def classify_venomous_snake(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make prediction using the pre-trained best model
        prediction = model_venomous.predict(x)[0]  # Get the prediction value from the list
        if prediction > 0.5:
            return True, prediction  # Return True if venomous snake detected along with confidence score
        else:
            return False, prediction  # Return False if non-venomous snake detected along with confidence score

    # Example usage:
    image_path = filepath
    snake_detected, confidence = detect_snakes_resnet(image_path)
    if snake_detected:
        print("Snake detected with confidence:", confidence)
        venomous_detected, venomous_confidence = classify_venomous_snake(image_path)
        if venomous_detected:

            # Make prediction
            prediction = model.predict(img_array)
            snake_probability = prediction[0][0]  # Probability of being a snake

            predicted_class = int(np.round(prediction[0][0]))  # Round to 0 or 1
            final_class = 'Venomous' if predicted_class == 1 else 'Non-Venomous'

            # Get probabilities for individual features
            eyes_percentage, head_percentage, tongue_percentage, pit_percentage, fang_percentage = prediction[1:]

            # Save the PIL image
            img.save('static/uploaded_image.jpg')

            return render_template('index.html', final_class=final_class, eyes_percentage=eyes_percentage,
                                head_percentage=head_percentage, tongue_percentage=tongue_percentage,
                                pit_percentage=pit_percentage, fang_percentage=fang_percentage, uploaded_image=True)
            #     print("Venomous snake detected with confidence:", venomous_confidence)
            # else:
            #     print("Non-venomous snake detected with confidence:", venomous_confidence)
    else:
        print("No snakes detected in the image.")
        return render_template('index.html', non_snake_error='The uploaded image does not contain a snake.')


    #  # Detect snakes using ResNet50
    # snake_detected = model_resnet(filepath)

    # if snake_detected:

    #     # Make prediction
    #     prediction = model.predict(img_array)
    #     snake_probability = prediction[0][0]  # Probability of being a snake

    #     predicted_class = int(np.round(prediction[0][0]))  # Round to 0 or 1
    #     final_class = 'Venomous' if predicted_class == 1 else 'Non-Venomous'

    #     # Get probabilities for individual features
    #     eyes_percentage, head_percentage, tongue_percentage, pit_percentage, fang_percentage = prediction[1:]

    #     # Save the PIL image
    #     img.save('static/uploaded_image.jpg')

    #     return render_template('index.html', final_class=final_class, eyes_percentage=eyes_percentage,
    #                         head_percentage=head_percentage, tongue_percentage=tongue_percentage,
    #                         pit_percentage=pit_percentage, fang_percentage=fang_percentage, uploaded_image=True)


    # else:
    #      return render_template('index.html', non_snake_error='The uploaded image does not contain a snake.')

        # # Define a threshold probability for classifying an image as a snake
        # snake_threshold = 0.001  # Adjust as needed

        # Check if the predicted probability of being a snake is above the threshold
        # if snake_probability < snake_threshold:

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
