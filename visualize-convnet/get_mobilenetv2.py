from keras.applications import MobileNetV2
import tensorflowjs as tfjs

model = MobileNetV2(weights='imagenet', alpha=0.35)

tfjs.converters.save_keras_model(model, 'mobilenetv2_0.35_tfjs')