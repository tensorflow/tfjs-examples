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

# Builds resources for the Iris demo.
# Note this is not necessary to run the demo, because we already provide hosted
# pre-built resources.
# Usage example: do this from the 'iris' directory:
#   ./build-resources.sh

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_EPOCHS=100
while true; do
  if [[ "$1" == "--epochs" ]]; then
    TRAIN_EPOCHS=$2
    shift 2
  elif [[ -z "$1" ]]; then
    break
  else
    echo "ERROR: Unrecognized argument: $1"
    exit 1
  fi
done

RESOURCES_ROOT="${DEMO_DIR}/dist/resources"
rm -rf "${RESOURCES_ROOT}"
mkdir -p "${RESOURCES_ROOT}"

# Run Python script to generate the pretrained model and weights files.
# Make sure you install the tensorflowjs pip package first.

python "${DEMO_DIR}/python/iris.py" \
    --epochs "${TRAIN_EPOCHS}" \
    --artifacts_dir "${RESOURCES_ROOT}"

cd ${DEMO_DIR}
yarn
yarn build

echo
echo "-----------------------------------------------------------"
echo "Resources written to ${RESOURCES_ROOT}."
echo "You can now run the demo with 'yarn watch'."
echo "-----------------------------------------------------------"
echo
