import tensorflow 
nm = tensorflow.keras.models.load_model('mod.h5')
print(nm.summary())
print('success!')