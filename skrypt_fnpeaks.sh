#!/bin/bash
OUTPUT_FILE="results_fnpeaks.txt"
> "$OUTPUT_FILE"


process_file() {
  dat_file="$1"
  ./fnpeaks_pro "$dat_file" 0.01 100 0.0001 | sed "s|$dat_file|$(basename "$dat_file")|" >> "$OUTPUT_FILE"
  echo "Przetwarzanie "$dat_file""
}

export OUTPUT_FILE
export -f process_file

find . -name "*.dat" | parallel -j$(nproc) process_file

echo "Przetwarzanie zakończone. Wyniki zapisano w $OUTPUT_FILE."
