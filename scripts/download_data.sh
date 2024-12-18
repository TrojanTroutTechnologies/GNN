#!/bin/bash
set -e
DATASET_NAME="WaterDrop"
OUTPUT_DIR="data/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in metadata.json train.tfrecord valid.tfrecord test.tfrecord
do
wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done
