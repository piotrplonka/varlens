#!/bin/bash
# Skrypt: clean_and_move_if_nan.sh

# Nazwa folderu docelowego
DEST_DIR="cleaned_files"

# Tworzenie folderu docelowego, jeśli nie istnieje
mkdir -p "$DEST_DIR"

# Pętla dla każdego pliku *.dat w bieżącym katalogu
for file in *.dat; do
    # Sprawdzenie, czy plik istnieje (gdyby nie było plików *.dat)
    [ -e "$file" ] || continue

    # Sprawdzenie, czy plik zawiera "nan" (ignorując wielkość liter)
    if grep -qi "nan" "$file"; then
        echo "Przetwarzam plik: $file"
        # Usunięcie wszystkich wierszy zawierających 'nan' (bez względu na wielkość liter)
        sed -i '/[nN][aA][nN]/d' "$file"
        # Przeniesienie pliku do folderu docelowego
        mv "$file" "$DEST_DIR"
    else
        echo "Plik $file nie zawiera 'nan' i pozostaje bez zmian."
    fi
done

echo "Operacja zakończona."

