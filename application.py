from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow 

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
	#Just for info basis
	# nm = tensorflow.keras.models.load_model('mod.h5')

	return render_template('main.html')


@app.route('/get_started')
def get_started():
	return render_template('get_started.html')


@app.route('/detect', methods = ['POST'])
def detect():
	#The main bread and butter of the api website!
	model = tensorflow.keras.models.load_model('mod.h5')


	raw_cxr = request.files['file']
	raw_cxr_name = raw_cxr.filename
	raw_cxr.save(raw_cxr_name)
	image = Image.open(raw_cxr_name).convert('L')
	# image = image.convert('RBG')
	cxr = image.resize((224, 224))
	cxr_np = np.asarray(cxr)
	if len(cxr_np.shape) == 2:
		cxr_np = np.stack((cxr_np,)*3, axis=-1)
		print(f'COnvt {cxr_np.shape} ')

	print('shape : ' + str(cxr_np.shape))
	cxr_np = cxr_np[...,:3]
	print("thy shape" + str(cxr_np.shape))
	cxr_np = cxr_np.reshape(1, 224, 224, 3) # image size feedabke to the neural network
	
	guess_normal = model.predict(cxr_np)[0]

	return jsonify({'guess_normal' : f'{guess_normal}' })


@app.route('/result', methods = ['POST'])
def result():
	model = tensorflow.keras.models.load_model('save_model.h5')


	raw_cxr = request.files['file']
	raw_cxr_name = raw_cxr.filename
	raw_cxr.save(raw_cxr_name)
	image = Image.open(raw_cxr_name).convert('L')
	# image = image.convert('RBG')
	cxr = image.resize((224, 224))
	cxr_np = np.asarray(cxr)
	if len(cxr_np.shape) == 2:
		cxr_np = np.stack((cxr_np,)*3, axis=-1)
		print(f'COnvt {cxr_np.shape} ')

	print('shape : ' + str(cxr_np.shape))
	cxr_np = cxr_np[...,:3]
	print("thy shape" + str(cxr_np.shape))
	cxr_np = cxr_np.reshape(1, 224, 224, 3) # image size feedabke to the neural network
	temp = model.predict(cxr_np/255.0)[0]
	print(temp)
	guess_covid = (temp[0])
	if guess_covid >=0.3:
		return render_template('result.html', message = "Likely Positive")
		# return render_template('result.html', message = "Not likely")
		
	# elif guess_covid <= 0.8:
	# 	return render_template('result.html', message = "Likely")
	else :
		return render_template('result.html', message = "Likely Negative")


	# return jsonify({'guess_normal' : f'{guess_normal}' })


if __name__ == '__main__':
	app.run(debug = True)



