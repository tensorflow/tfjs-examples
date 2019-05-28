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

# Builds resources for the sequence-to-sequence English-French translation demo.
# Note this is not necessary to run the demo, because we already provide hosted
# pre-built resources.
# Usage example: do this in the 'translation' directory:
#   ./build.sh ~/ml-data/fra-eng/fra.txt
#
# You can specify the number of training epochs by using the --epochs flag.
# For example:
#   ./build-resources.sh ~/ml-data/fra-eng/fra.txt --epochs 10

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_DATA_PATH="$1"
if [[ -z "${TRAIN_DATA_PATH}" ]]; then
  echo "ERROR: TRAIN_DATA_PATH is not specified."
  echo "You can download the training data with a command such as:"
  echo "  wget http://www.manythings.org/anki/fra-eng.zip"
  exit 1
fi
shift 1

if [[ ! -f ${TRAIN_DATA_PATH} ]]; then
  echo "ERROR: Cannot find training data at path '${TRAIN_DATA_PATH}'"
  exit 1
fi

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

python "${DEMO_DIR}/python/translation.py" \
    "${TRAIN_DATA_PATH}" \
    --recurrent_initializer orthogonal \
    --artifacts_dir "${RESOURCES_ROOT}" \
    --epochs "${TRAIN_EPOCHS}"

cd ${DEMO_DIR}
yarn
yarn build

echo
echo "-----------------------------------------------------------"
echo "Resources written to ${RESOURCES_ROOT}."
echo "You can now run the demo with 'yarn watch'."
echo "-----------------------------------------------------------"
echo
