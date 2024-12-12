#!/bin/bash

OUTPUT_FILE="results_AOV.txt"

> "$OUTPUT_FILE"

export OUTPUT_FILE

find . -maxdepth 1 -name "*.dat" | parallel -j$(nproc) '
  filename=$(basename {})
  ./AOV_pro {} | sed "s|{}|$filename|" | tee -a "$OUTPUT_FILE"
'
echo "Przetwarzanie zakończone. Wyniki zapisano w $OUTPUT_FILE."

