#!/bin/bash
OUTPUT_FILE="results_AOV.txt"
> "$OUTPUT_FILE"


process_file() {
  dat_file="$1"
  ./AOV_pro "$dat_file"| sed "s|$dat_file|$(basename "$dat_file")|" >> "$OUTPUT_FILE"
  echo "Przetwarzanie "$dat_file""
}

export OUTPUT_FILE
export -f process_file

find . -name "*.dat" | parallel -j$(nproc) process_file

echo "Przetwarzanie zakończone. Wyniki zapisano w $OUTPUT_FILE."

