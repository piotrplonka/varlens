#!/bin/bash

OUTPUT_FILE="results_fnpeaks.txt"

> "$OUTPUT_FILE"

export OUTPUT_FILE

find "$DATA_DIR" -name "*.dat" | parallel -j$(nproc) '
  filename=$(basename {})
  ./fnpeaks_pro {} 0.01 100 0.0001 | sed "s|{}|$filename|" | tee -a "$OUTPUT_FILE"

echo "Przetwarzanie zakończone. Wyniki zapisano w $OUTPUT_FILE."

