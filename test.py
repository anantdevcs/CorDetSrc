import tensorflow 
import cv2
import os
nm = tensorflow.keras.models.load_model('save_model.h5')
print(nm.summary())
print("For covid positive cases train")
test_path_pos = './train/positive'
for img in os.listdir(test_path_pos):
	fp  =test_path_pos + '/' + img
	img = cv2.imread(fp)
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
	resized = resized.reshape(1,224,224,3)/255.0
	print(nm.predict(resized) )

print("For covid negative cases train")
test_path_pos = './train/negative'
for img in os.listdir(test_path_pos):
	fp  =test_path_pos + '/' + img
	img = cv2.imread(fp)
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
	resized = resized.reshape(1,224,224,3)/255.0
	print(nm.predict(resized) )

print("For covid positive cases val")
test_path_pos = './val/positive'
for img in os.listdir(test_path_pos):
	fp  =test_path_pos + '/' + img
	img = cv2.imread(fp)
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
	resized = resized.reshape(1,224,224,3)/255.0
	print(nm.predict(resized) )

print("For covid negative cases val")
test_path_pos = './val/negative'
for img in os.listdir(test_path_pos):
	fp  =test_path_pos + '/' + img
	img = cv2.imread(fp)
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
	resized = resized.reshape(1,224,224,3)/255.0
	print(nm.predict(resized) )


print("For covid positive cases test")
test_path_pos = './test/positive'
for img in os.listdir(test_path_pos):
	fp  =test_path_pos + '/' + img
	img = cv2.imread(fp)
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
	resized = resized.reshape(1,224,224,3)/255.0
	print(nm.predict(resized) )

print("For covid negative cases test")
test_path_pos = './test/negative'
for img in os.listdir(test_path_pos):
	fp  =test_path_pos + '/' + img
	img = cv2.imread(fp)
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
	resized = resized.reshape(1,224,224,3)/255.0
	print(nm.predict(resized) )





print('success!')