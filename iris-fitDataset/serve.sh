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

# This script starts two HTTP servers on different ports:
#  * Port 1234 (using parcel) serves HTML and JavaScript.
#  * Port 1235 (using http-server) serves pretrained model resources.
#
# The reason for this arrangement is that Parcel currently has a limitation that
# prevents it from serving the pretrained models; see
# https://github.com/parcel-bundler/parcel/issues/1098.  Once that issue is
# resolved, a single Parcel server will be sufficient.

NODE_ENV=development
RESOURCE_PORT=1235

# Ensure that http-server is available
yarn

echo Starting the pretrained model server...
node_modules/http-server/bin/http-server dist --cors -p "${RESOURCE_PORT}" > /dev/null & HTTP_SERVER_PID=$!

echo Starting the example html/js server...
# This uses port 1234 by default.
parcel index.html --dist-dir dist --open --no-hmr --public-url / -p 1236

# When the Parcel server exits, kill the http-server too.
kill $HTTP_SERVER_PID
