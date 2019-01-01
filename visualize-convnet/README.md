# TensorFlow.js Example: Visualizing Convnet Filters

TODO(cais):

## How to use this demo

TODO(cais): Simplify the below.

```sh
pip install -r requirements.txt
python get_vgg16.py
yarn visualize ./vgg16_tfjs/model.json block1_conv1,block2_conv1,block3_conv1,block4_conv1,block5_conv1 --filters 64 --gpu --outputDir dist/filters
yarn visualize ./vgg16_tfjs/model.json block1_conv1,block2_conv1,block3_conv1,block4_conv1,block5_conv1 --inputImage cat.jpg --gpu --filters 64 --outputDir dist/activation
```