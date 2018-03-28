#!/usr/bin/env bash

# Copyright 2018 Google LLC. All Rights Reserved.
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

# Builds the Sentiment demo for TensorFlow.js Layers.
# Usage example: do this from the 'sentiment' directory:
#   ./build.sh lstm
#
# Then open the demo HTML page in your browser, e.g.,
#   http://localhost:8000

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  build-imdb-demo.sh <MODEL_TYPE>"
  echo
  echo "MODEL_TYPE options: lstm | cnn"
  exit 1
fi
MODEL_TYPE=$1
shift
echo "Using model type: ${MODEL_TYPE}"

DEMO_PORT=8000
TRAIN_EPOCHS=5
while true; do
  if [[ "$1" == "--port" ]]; then
    DEMO_PORT=$2
    shift 2
  elif [[ "$1" == "--epochs" ]]; then
    TRAIN_EPOCHS=$2
    shift 2
  elif [[ -z "$1" ]]; then
    break
  else
    echo "ERROR: Unrecognized argument: $1"
    exit 1
  fi
done

DATA_ROOT="${DEMO_DIR}/dist/data"
rm -rf "${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

# Run Python script to generate the pretrained model and weights files.
# Make sure you install the tensorflowjs pip package first.

python "${DEMO_DIR}/python/imdb.py" \
    "${MODEL_TYPE}" \
    --epochs "${TRAIN_EPOCHS}" \
    --artifacts_dir "${DATA_ROOT}"

cd ${DEMO_DIR}
yarn
yarn build

echo
echo "-----------------------------------------------------------"
echo "Once the HTTP server has started, you can view the demo at:"
echo "  http://localhost:${DEMO_PORT}"
echo "-----------------------------------------------------------"
echo

node_modules/http-server/bin/http-server ./dist -p "${DEMO_PORT}"
