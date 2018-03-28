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

# Builds the Iris demo for TensorFlow.js Layers.
# Usage example: do this from the 'iris' directory:
#   ./scripts/build-iris-demo.sh
#
# Then open the demo HTML page in your browser, e.g.,
#   http://localhost:8000/dist

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DEMO_DIR}/dist/data"
rm -rf "${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

# Run Python script to generate the pretrained model and weights files.
# Make sure you install the tensorflowjs pip package first.

python "${DEMO_DIR}/python/iris.py" --artifacts_dir "${DATA_ROOT}"

cd ${DEMO_DIR}
yarn
yarn build

echo
echo "-----------------------------------------------------------"
echo "Once the HTTP server has started, you can view the demo at:"
echo "  http://localhost:${DEMO_PORT}/dist"
echo "-----------------------------------------------------------"
echo

node_modules/http-server/bin/http-server -p "${DEMO_PORT}"
