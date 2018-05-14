#!/bin/sh
#=============================================================================
# @license
# Copyright 2018 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Run this script via yarn: `yarn download-data`

PITCH_TYPE_TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_training_data.json.gz"
PITCH_TYPE_TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_test_data.json.gz"
PITCH_TYPE_VALIDATION_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_validation_data.json.gz"

STRIKE_ZONE_TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/strike_zone_training_data.json.gz"
STRIKE_ZONE_TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/strike_zone_test_data.json.gz"

mkdir -p dist/

curl -L $PITCH_TYPE_TRAINING_DATA | gunzip > dist/pitch_type_training_data.json
curl -L $PITCH_TYPE_TEST_DATA | gunzip > dist/pitch_type_test_data.json
curl -L $PITCH_TYPE_VALIDATION_DATA | gunzip > dist/pitch_type_validation_data.json

curl -L $STRIKE_ZONE_TRAINING_DATA | gunzip > dist/strike_zone_training_data.json
curl -L $STRIKE_ZONE_TEST_DATA | gunzip > dist/strike_zone_test_data.json
