#!/bin/bash
# workaround for https://github.com/gitpod-io/gitpod/issues/596
URL=$1
PORT=${URL##*:}
PUB=$(gp url $PORT)
gp preview $PUB