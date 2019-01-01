from keras.applications.vgg16 import VGG16
import tensorflowjs as tfjs

model = VGG16(weights='imagenet')

tfjs.converters.save_keras_model(model, 'vgg16_tfjs')
