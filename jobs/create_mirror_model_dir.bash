#!/bin/bash

set -o pipefail
set -o errexit
set -o nounset

BASE_MODEL_NAME="$1"
NEW_MODEL_NAME="$2"

if [ -z "$BASE_MODEL_NAME" ] || [ -z "$NEW_MODEL_NAME" ]; then
    echo "Usage: bash $0 [BASE_MODEL_NAME] [NEW_MODEL_NAME]"
    exit 1
fi

BASE_MODEL_DIR="data/$BASE_MODEL_NAME"
NEW_MODEL_DIR="data/$NEW_MODEL_NAME"

if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "Error: Base model directory '$BASE_MODEL_DIR' does not exist."
    exit 1
fi

# Copy base config to new model directory
mkdir -p "$NEW_MODEL_DIR"
cp "$BASE_MODEL_DIR/config.yaml" "$NEW_MODEL_DIR/config.yaml"

# Open nano to edit the new config file
nano "$NEW_MODEL_DIR/config.yaml"

# ask for it
read -p "Do you want to create a symbolic link for the webdataset? (y/n): " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    if [ ! -d "$BASE_MODEL_DIR/webdataset" ]; then
        echo "Error: Base model webdataset directory '$BASE_MODEL_DIR/webdataset' does not exist."
        exit 1
    fi
    ln -s "$(realpath "$BASE_MODEL_DIR/webdataset")" "$NEW_MODEL_DIR/webdataset"
    echo "Symbolic link created: $NEW_MODEL_DIR/webdataset -> $BASE_MODEL_DIR/webdataset"
else
    echo "Skipping symbolic link creation. You can set up the webdataset directory manually later."
fi

echo "Set up ${NEW_MODEL_DIR} by copying config from ${BASE_MODEL_DIR}"
