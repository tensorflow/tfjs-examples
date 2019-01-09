#!/usr/bin/env bash
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# This is the driving script of the visualize-convnet example.
#
# Syntax:
#   main.sh [--image <PATH_TO_IMAGE_FILE>] [--filters <NUM_FILTERS>] [--gpu]
#
# Argumnets:
#   --image <PATH_TO_IMAGE_FILE>
#     Path to image file. If not specified, defaults to "cat.jpg".
#
#   --filters <NUM_FILTERS>
#     Number of filters to visualize. Default: 8
#
#   --gpu
#     Use tfjs-node-gpu instead of the default tfjs-node (required CUDA GPU)

set -e

# Parse input arguments 
IMAGE="cat.jpg"
FILTERS="8"
GPU_FLAG=""
while [[ ! -z "$1" ]]; do
  if [[ "$1" == "--image" ]]; then
    IMAGE="$2"
    shift 2
  elif [[ "$1" == "--filters" ]]; then
    FILTERS="$2"
    shift 2
  elif [[ "$1" == "--gpu" ]]; then
    GPU_FLAG="--gpu"
    shift
  else
    echo "ERROR: Unrecognized option: ${$1}"
    exit 1
  fi
done

echo "Installing Python libraries..."
pip install -r requirements.txt

if [[ ! -f "vgg16_tfjs/model.json" ]]; then
  echo "Downloading and converting VGG16 model..."
  python get_vgg16.py
else
  echo "VGG16 model has already been downloaded and converted."
fi

yarn

# Clean up old files.
rm -rf dist/activation dist/filters

echo "Calculating maximally-activating input images for convnet filters..."
LAYER_NAMES="block1_conv1,block2_conv1,block3_conv2,block4_conv2"
node main.js \
    "./vgg16_tfjs/model.json" \
    "${LAYER_NAMES}" \
    --filters "${FILTERS}" ${GPU_FLAG} \
    --outputDir dist/filters

echo "Calculating convnet activations and class activation map (CAM)..."
LAYER_NAMES="block1_conv1,block2_conv1,block3_conv2,block4_conv2,block5_conv3"
node main.js \
    "./vgg16_tfjs/model.json" \
    "${LAYER_NAMES}" \
    --filters "${FILTERS}" ${GPU_FLAG} \
    --inputImage "${IMAGE}" \
    --outputDir dist/activation

echo "Launching parcel server and opening page in browser..."
yarn watch
