# https://keras.io/zh/applications/
from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
model = ResNet50V2(weights='imagenet')
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
# Predicted: [(u'n02124075', u'Egyptian_cat', 0.4909497), (u'n02123597', u'Siamese_cat', 0.3463773), (u'n02123045', u'tabby', 0.10439434)]
