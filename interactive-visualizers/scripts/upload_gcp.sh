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

set -e

# get packages directories
VISUALIZER_PACKAGE_DIR="dist/interactive-visualizers/*"
PLAYGROUND_PACKAGE_DIR="dist/playground/*"
# host url
PACKAGE_HOST="interactive_visualizer"

#rm -rf dist/
#yarn prod-build
#yarn playground-prod-build

PACKAGE_VERSION=`node -p "require('./package.json').version"`
echo 'current version: ' $PACKAGE_VERSION
# remove the pre-built addon tarball if it already exist
if [ "$1" = "for-publish" ]; then
  echo 'copying ...'
  gsutil -m cp $VISUALIZER_PACKAGE_DIR gs://$PACKAGE_HOST/$PACKAGE_VERSION
  gsutil -m cp $PLAYGROUND_PACKAGE_DIR gs://$PACKAGE_HOST/$PACKAGE_VERSION/playground
fi
