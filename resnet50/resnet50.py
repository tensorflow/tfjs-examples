# https://keras.io/zh/applications/
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
model = ResNet50(weights='imagenet')
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
tfjs.converters.save_keras_model(model, 'resnet_model_json')
# (class, description, probability)
#
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02124075', u'Egyptian_cat', 0.8075683), (u'n02123597', u'Siamese_cat', 0.06937122), (u'n02123045', u'tabby', 0.051120896)]
