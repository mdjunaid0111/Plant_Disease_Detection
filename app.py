import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Define upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def getResult(image_path):
    """Process the image and get predictions."""
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        return "No file part", 400
    
    f = request.files['file']
    
    if f.filename == '':
        return "No selected file", 400

    if f:
        # Secure and construct the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        print(f"Saving file to: {file_path}")
        
        # Save the file
        f.save(file_path)
        
        # Ensure the file was saved
        if not os.path.isfile(file_path):
            return "File not found after saving.", 400
        
        # Get the prediction result
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    
    return "File upload failed", 400

if __name__ == '__main__':
    app.run(debug=True)
