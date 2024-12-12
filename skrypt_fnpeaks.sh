#!/bin/bash

OUTPUT_FILE="results.txt"

> "$OUTPUT_FILE"

export OUTPUT_FILE

find "$DATA_DIR" -name "*.dat" | parallel -j$(nproc) '
  filename=$(basename {})
  ./fnpeaks_pro {} 0.01 250 0.001 | sed "s|{}|$filename|" | tee -a "$OUTPUT_FILE"

echo "Przetwarzanie zakończone. Wyniki zapisano w $OUTPUT_FILE."

