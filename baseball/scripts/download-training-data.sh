#!/bin/sh

PITCH_TYPE_TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_training_data.json.gz"
PITCH_TYPE_TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_test_data.json.gz"

curl -L $PITCH_TYPE_TRAINING_DATA | gunzip > pitch_type_training_data.json
curl -L $PITCH_TYPE_TEST_DATA | gunzip > pitch_type_test_data.json