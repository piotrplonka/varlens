#!/bin/bash

# Katalog z programem i główny katalog ews
base_dir="/home/pplonka/Desktop/varlens"
ews_dir="$base_dir/ews"
in_dir="$base_dir/In"
out_dir="$base_dir/Out"

# Skrypt Pythonowy
python_script="$base_dir/cut_on_2_files.py"

# Utwórz katalogi In i Out jeśli nie istnieją
mkdir -p "$in_dir" "$out_dir"

# Maksymalna liczba procesów równoległych
max_jobs=100
current_jobs=0

# Funkcja do przetwarzania folderu
process_folder() {
    folder=$1
    echo "Przetwarzanie folderu: $folder"
    
    # Sprawdzamy, czy folder zawiera pliki 'params.dat' i 'phot.dat'
    if [[ -f "$folder/params.dat" && -f "$folder/phot.dat" ]]; then
        # Przenosimy skrypt Pythonowy do folderu
        echo "Przenoszenie skryptu Pythonowego do folderu: $folder"
        cp "$python_script" "$folder"

        # Przechodzimy do folderu, w którym uruchamiamy skrypt
        cd "$folder" || exit 1

        # Uruchamiamy skrypt Pythonowy dla tego folderu
        echo "Uruchamianie skryptu Pythonowego..."
        python3 "$folder/cut_on_2_files.py" "$folder"

        # Usuwamy skrypt Pythonowy z folderu po wykonaniu
        rm "$folder/cut_on_2_files.py"
        echo "Skrypt Pythonowy został usunięty z folderu."

        # Przenosimy wszystkie pliki *in.dat i *out.dat do odpowiednich folderów
        for file in "$folder"/*in.dat; do
            if [[ -f "$file" ]]; then
                mv "$file" "$in_dir"
                echo "Przeniesiono $file do $in_dir"
            fi
        done

        for file in "$folder"/*out.dat; do
            if [[ -f "$file" ]]; then
                mv "$file" "$out_dir"
                echo "Przeniesiono $file do $out_dir"
            fi
        done

    else
        echo "Brak wymaganych plików w folderze: $folder"
    fi
}

# Funkcja do uruchomienia procesu w równoległości dla każdego folderu
process_year() {
    year_dir=$1
    for folder in "$year_dir"/*; do
        if [[ -d "$folder" ]]; then
            # Uruchamiamy proces w tle
            process_folder "$folder" &
            current_jobs=$((current_jobs+1))
            
            # Sprawdzamy, czy liczba procesów przekroczyła dozwolony limit
            if [[ $current_jobs -ge $max_jobs ]]; then
                # Czekamy na zakończenie jednego z procesów
                wait -n
                current_jobs=$((current_jobs-1))
            fi
        fi
    done
}

export -f process_folder

# Przechodzimy po wszystkich latach w katalogu 'ews' i równolegle przetwarzamy foldery
for year_dir in "$ews_dir"/*; do
    if [[ -d "$year_dir" ]]; then
        process_year "$year_dir"
    fi
done

# Czekamy na zakończenie wszystkich procesów równoległych
wait

echo "Przetwarzanie zakończone."

