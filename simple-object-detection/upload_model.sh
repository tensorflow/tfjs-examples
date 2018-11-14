#!/usr/bin/env bash
#
# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Uploads trained model to centralized storage location (in GCS).
#
# Before you run this script, make sure you have
#   1. Installed gsutil (https://cloud.google.com/storage/docs/gsutil_install)
#   2. Authenticated and initiated gsutil.
# 
# Usage:
#   upload_model.sh

set -e

SRC_PATH="dist/object_detection_model"
DEST_GS_URL="gs://tfjs-examples/simple-object-detection/dist"

gsutil cp -r "${SRC_PATH}" "${DEST_GS_URL}"
