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

# Quantize a specified model saved from the command `yarn train` and evaluates
# the test accuracy under different levels of quantization (8-bit and 16-bit).

set -e

MODEL_NAME=$1
if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: quantize_evaluate <MODEL_NAME>"
  exit 1
fi

# Make sure model is available.
MODEL_ROOT="models/${MODEL_NAME}"
MODEL_PATH="${MODEL_ROOT}/original"
MODEL_JSON_PATH="${MODEL_PATH}/model.json"

# Make sure pip3 is available.
if [[ -z "$(which pip3)" ]]; then
  echo "ERROR: Cannot find pip3 on path."
  echo "       Make sure you have python and pip3 installed."
  exit 1
fi

if [[ -z "$(which virtualenv)" ]]; then
  echo "Installing virtualenv..."
  pip3 install virtualenv
fi

VENV_DIR="$(mktemp -d)_venv"
echo "Creating virtualenv at ${VENV_DIR} ..."
virtualenv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip3 install tensorflowjs

if [[ "${MODEL_NAME}" == "MobileNetV2" ]]; then
  # Save the MobilNetV2 model first.
  if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
    python save_mobilenetv2.py
  fi
fi

if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
  echo "ERROR: Cannot find model JSON file at ${MODEL_JSON_PATH}"
  echo "       Make sure you train and save a model with the"
  echo "       following command first: yarn train"
  rm -rf "${VENV_DIR}"
  exit 1
fi

# Perform 16-bit quantization.
MODEL_PATH_16BIT="${MODEL_ROOT}/quantized-16bit"
rm -rf "${MODEL_PATH_16BIT}"
tensorflowjs_converter \
    --input_format tfjs_layers_model \
    --output_format tfjs_layers_model \
    --quantization_bytes 2 \
    "${MODEL_JSON_PATH}" "${MODEL_PATH_16BIT}"

# Perform 8-bit quantization.
MODEL_PATH_8BIT="${MODEL_ROOT}/quantized-8bit"
rm -rf "${MODEL_PATH_8BIT}"
tensorflowjs_converter \
    --input_format tfjs_layers_model \
    --output_format tfjs_layers_model \
    --quantization_bytes 1 \
    "${MODEL_JSON_PATH}" "${MODEL_PATH_8BIT}"

# Clean up the virtualenv
rm -rf "${VENV_DIR}"

yarn

if [[ "${MODEL_NAME}" == "MobileNetV2" ]]; then
  # Download the data required for evaluating MobileNetV2.
  IMAGENET_1000_SAMPLES_DIR="imagenet-1000-samples"

  if [[ ! -d "${IMAGENET_1000_SAMPLES_DIR}" ]]; then
    curl -o imagenet-1000-samples.tar.gz \
        https://storage.googleapis.com/tfjs-examples/quantization/data/imagenet-1000-samples.tar.gz
    mkdir -p ${IMAGENET_1000_SAMPLES_DIR}
    tar xf imagenet-1000-samples.tar.gz
    rm imagenet-1000-samples.tar.gz
  fi

  # Evaluate accuracy under no quantization (i.e., full 32-bit weight precision).
  echo "=== Accuracy evalution: No quantization ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_JSON_PATH}" \
      "${IMAGENET_1000_SAMPLES_DIR}"


  # Evaluate accuracy under 16-bit quantization.
  echo "=== Accuracy evalution: 16-bit quantization ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_16BIT}/model.json" \
      "${IMAGENET_1000_SAMPLES_DIR}"

  # Evaluate accuracy under 8-bit quantization.
  echo "=== Accuracy evalution: 8-bit quantization ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_8BIT}/model.json" \
      "${IMAGENET_1000_SAMPLES_DIR}"
else
  # Evaluate accuracy under no quantization (i.e., full 32-bit weight precision).
  echo "=== Accuracy evalution: No quantization ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_JSON_PATH}"

  # Evaluate accuracy under 16-bit quantization.
  echo "=== Accuracy evalution: 16-bit quantization ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_16BIT}/model.json"

  # Evaluate accuracy under 8-bit quantization.
  echo "=== Accuracy evalution: 8-bit quantization ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_8BIT}/model.json"
fi

function calc_gzip_ratio() {
  ORIGINAL_FILES_SIZE_BYTES="$(ls -lAR ${1} | grep -v '^d' | awk '{total += $5} END {print total}')"
  TEMP_TARBALL="$(mktemp)"
  tar czf "${TEMP_TARBALL}" "${1}"
  TARBALL_SIZE="$(wc -c < ${TEMP_TARBALL})"
  ZIP_RATIO="$(awk "BEGIN { print(${ORIGINAL_FILES_SIZE_BYTES} / ${TARBALL_SIZE}) }")"
  rm "${TEMP_TARBALL}"

  echo "  Total file size: ${ORIGINAL_FILES_SIZE_BYTES} bytes"
  echo "  gzipped tarball size: ${TARBALL_SIZE} bytes"
  echo "  gzip ratio: ${ZIP_RATIO}"
  echo
}

echo
echo "=== gzip ratios ==="

# Calculate the gzip ratio of the original (unquantized) model.
echo "Original model (No quantization):"
calc_gzip_ratio "${MODEL_PATH}"

# Calculate the gzip ratio of the 16-bit-quantized model.
echo "16-bit-quantized model:"
calc_gzip_ratio "${MODEL_PATH_16BIT}"

# Calculate the gzip ratio of the 8-bit-quantized model.
echo "8-bit-quantized model:"
calc_gzip_ratio "${MODEL_PATH_8BIT}"
