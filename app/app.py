from flask import Flask, request, render_template, redirect, url_for # Import to do all routing stuff
# from app import app # Import app to use routes 
import stripe # Import for payment process
import os # Import to get files
from dotenv import load_dotenv # Import to load production.env file
# from .ml_sid import * # Import to build the model and predict a image
import tensorflow as tf # Import to build the graph for the model

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json

from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')

def build_model():
	with open('multi_disease_model.json', 'r') as json_file:
		architecture = json.load(json_file)
		model = model_from_json(json.dumps(architecture))

	model.load_weights('multi_disease_model_weight.h5')
	# model._make_predict_function()
	return model


def load_image(img_path):
	print('---- loading the image -----')
	print(img_path)
	img = image.load_img(img_path, target_size=(128, 128, 3))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img /= 255.
	print('-------- image loaded ----------')
	return img


def predict_image(model, img_path, biggest_result=False, show_result=False):
  new_image = load_image(img_path)
  print('-------- passing image to model ----------')
  pred = model.predict(new_image)

  if not show_result:
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img, cmap='bone')
    plt.title(pred)
    plt.show()

  return (np.argmax(pred), np.max(pred)) if biggest_result else pred

model = build_model() # Build the model with the specific json and h5 weights file
graph = tf.compat.v1.get_default_graph() # Get the default graph from tensorflow

# Load Environment Variables File
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, './production.env'))

# Get the pub and secret key from the env file and set the stripe api key
pub_key = os.getenv('pub_key')
secret_key = os.getenv('secret_key')
stripe.api_key = secret_key

# Define the allowed file extensions that can be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Defining the model labels
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
					'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
					'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


@app.route('/')
def index():
	return render_template('index.html', pub_key=pub_key)


@app.route('/image_upload')
def image_upload():
	return render_template('image_upload.html', predictions=[])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST', 'GET'])
def predict():
	print('-------- prediction started ----------')
	if 'file' not in request.files:
		return render_template('image_upload.html', predictions=[])
	
	file = request.files['file']
	print('-------- file.filename is printed ----------')
	print(file.filename)
	if file.filename == '':
		return render_template('image_upload.html', predictions=[])
	
	if file and allowed_file(file.filename):
		print('--------- inside if --------')
		filename = secure_filename(file.filename)
		file.save(file.filename)
		print('------ file saved ------')
		global graph
		# with graph.as_default():
		predictions = predict_image(model, file.filename)
		print('-------- prediction done ----------')
		os.remove(file.filename)
		print(list(predictions[0]))
		return render_template('image_upload.html', predictions=list(predictions[0]))

	return render_template('image_upload.html', predictions=[])


@app.route('/pay', methods=['POST', 'GET'])
def pay():
	return redirect(url_for('image_upload'))

