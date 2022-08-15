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

# This script tests various yarn commands in the quantization example.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Wipe all models to start from a clean state for testing.
if [[ -d "models" ]]; then
  echo "ERROR: models/ directory exists. Make sure you remove it or "
  echo "       move it to a different path before running this testing script."
  exit 1
fi

if [[ -z "$(which pip3)" ]]; then
  echo "pip3 is not on path. Attempting to install it..."
  apt-get update
  apt-get install -y python3-pip
fi

yarn

# Call housing model training script.
yarn train-housing --epochs 1

# Verify that the housing model has been saved.
HOUSING_MODEL_JSON_PATH="models/housing/original/model.json"
if [[ ! -f "${HOUSING_MODEL_JSON_PATH}" ]]; then
  echo "ERROR: Failed to find expected model.json at ${HOUSING_MODEL_JSON_PATH}"
  exit 1
fi

# Call housing model eval script
yarn quantize-and-evaluate-housing

# Call Fashion-MNIST training script.
# An epoch of Fashion-MNIST model taks too long. Set it to 0 epochs.
# The model should be constructed and saved to disk nonetheless.
yarn train-fashion-mnist --epochs 0

# Verify that the housing model has been saved.
FASHION_MNIST_MODEL_JSON_PATH="models/fashion-mnist/original/model.json"
if [[ ! -f "${FASHION_MNIST_MODEL_JSON_PATH}" ]]; then
  echo "ERROR: Failed to find expected model.json at ${FASHION_MNIST_MODEL_JSON_PATH}"
  exit 1
fi

yarn quantize-and-evaluate-fashion-mnist

# Call MNIST training script.
# An epoch of MNIST model taks too long. Set it to 0 epochs.
# The model should be constructed and saved to disk nonetheless.
yarn train-mnist --epochs 0

# Verify that the housing model has been saved.
MNIST_MODEL_JSON_PATH="models/mnist/original/model.json"
if [[ ! -f "${MNIST_MODEL_JSON_PATH}" ]]; then
  echo "ERROR: Failed to find expected model.json at ${MNIST_MODEL_JSON_PATH}"
  exit 1
fi

yarn quantize-and-evaluate-mnist

# Clean up models/ directory
rm -rf models/
