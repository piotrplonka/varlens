#!/bin/bash
# Sprawdzenie, czy podano argument (plik do przetworzenia)
if [ "$#" -ne 1 ]; then
    echo "Użycie: $0 plik.txt"
    exit 1
fi

plik="$1"

# Przetwarzamy plik przy użyciu awk:
# - $NF oznacza ostatnią kolumnę
# - Jeśli ostatnia kolumna jest równa "ID=PHYSICAL", "ID=ECL" lub "ID=?"
# - !seen[$1]++ powoduje, że dla każdej unikalnej wartości pierwszej kolumny drukujemy tylko pierwszy napotkany wiersz
awk '($NF=="ID=PHYSICAL" || $NF=="ID=ECL" || $NF=="ID=?") && !seen[$1]++ { print }' "$plik"

