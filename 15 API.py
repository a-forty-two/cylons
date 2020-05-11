from keras.applications import ResNet50 # IMAGE: VGG, Resnet, Alexnet 
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils 
from PIL import Image 
import numpy as np
import flask
import io 
import tensorflow as tf
graph = tf.get_default_graph()

app = flask.Flask(__name__)
model = None

def load_model():
	global model
	model = ResNet50(weights='imagenet') 

# LOAD MODEL
# CLEAN DATA
# define a route for model.predict



def processImage(image, target):
	if image.mode != 'RGB':
		image= image.convert('RGB')
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
	return image

# ImageDataGenerator(Augmentator) -> VGG
# Imagenet_utils -> ResNET 

# FIRST : DEFINE a failure JSON, IFF everything goes well, then make success TRUE
# from  a POST method, collect the file, record, key-value or data that was input
# then on this input, apply data cleaning functions-> EXACTLY in the same
# sequence as that of training the model!
# after cleaning-> EXPAND dims on axis=0 to convert your data into sequential data
# then make predictions! 


@app.route('/predict', methods=['POST'] )
def predict():
	data = {"success":False}
	if flask.request.method == 'POST':
		if flask.request.files.get('image'):
			img = flask.request.files['image'].read()
			image = Image.open(io.BytesIO(img))
			image = processImage(image, target=(224,224))	
			preds = []
			
			preds = model.predict(image)
			# reversing the binarized labels preds
	
			results = imagenet_utils.decode_predictions(preds)
			data['predictions'] = []
			for(imagenetId, label, prob) in results[0]:
				rslt = {"label":label, "confidence":float(prob) }
				data['predictions'].append(rslt)
			data['success'] = True
	return flask.jsonify(data) 

if __name__ == '__main__':
	load_model()
	app.run()





