#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import math
import multiprocessing

filename_data = 'phot.dat'
data = np.loadtxt(filename_data, usecols=(0, 1, 2))
time, magnitudo, error = data[:, 0], data[:, 1], data[:, 2]

flux = 10**(-0.4 * (magnitudo-22))
flux_error = 0.4*np.log(10)*error #10**(-0.4 * (error))*f# 0.4*np.log(10)*error


sigma=np.sqrt(np.var(flux))

# Poszukiwanie maksimum jasności jako mediany 10 najjaśniejszych punktów
sorted_indices = np.argsort(magnitudo)
lowest_magnitudes_indices = sorted_indices[:10]
lowest_magnitudes_time = time[lowest_magnitudes_indices]
median_time = np.median(lowest_magnitudes_time)

#Baseline początkowa wartość
sorted_indices_flux = np.argsort(flux)
lowest_flux = sorted_indices[:50]

baseline_guess_flux=np.mean(lowest_flux)#np.median(flux)
#print(baseline_guess_flux)

# Funkcja chi-kwadrat 
def chi2_fun(params, time, flux, flux_error):
    t_0, u_0, t_E, baseline_source, baseline_blending = params
    # Sprawdzanie ograniczeń na u_0
    if u_0 < 0 or u_0 > 1 or t_E < 0 or baseline_blending < 0 or baseline_source < 0 or t_E > 1000: #or (baseline_blending+baseline_source) > 1.1*baseline_guess_flux:
        return np.inf  

    u = np.sqrt(u_0**2 + ((time - t_0) / t_E)**2)
    A = ((u**2 + 2) / (u * np.sqrt(u**2 + 4)))
    flux_calculated = baseline_source * A + baseline_blending
    chi2 = np.sum(((flux - flux_calculated) / flux_error)**2)
    return chi2


# Proces dopasowania modelu 
def process_model(guess, time, flux, flux_error):
    result = op.minimize(
        chi2_fun, x0=guess, args=(time, flux, flux_error), method='Nelder-Mead', 
        options={
            'xatol': 1e-6,   # Dokładność w parametrach
            'fatol': 1e-6,   # Dokładność w funkcji celu
            'maxiter': 5000  # Zwiększenie liczby iteracji
        }
    )
    if result.success:
        return result.x, result.fun
    return None, np.inf

# Funkcja worker (MUSI BYĆ globalna, nie wewnątrz funkcji `find_best_model_parallel`)
def worker(args):
    guess, time, flux, flux_error = args
    return process_model(guess, time, flux, flux_error)

# Poszukiwanie najlepszego modelu z wieloma wątkami
def find_best_model_parallel(initial_guesses, time, flux, flux_error, num_processes=12):
    best_result = None
    best_chi2 = np.inf

    # Przygotowanie danych do wielowątkowości
    tasks = [(guess, time, flux, flux_error) for guess in initial_guesses]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker, tasks)

    for params, chi2 in results:
        print(chi2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_result = params

    return best_result, best_chi2


guess_u_0 = np.linspace(0.01, 1, 10)
guess_t_E = np.linspace(1, 200, 10)

# Stałe wartości dla pozostałych parametrów
fixed_t_0 = median_time  
fixed_source = np.median(flux) * 0.5  
fixed_blend = np.median(flux) * 0.5  

# Iteracja tylko po u_0 i t_E
initial_guesses = [
    (fixed_t_0, u, t, fixed_source, fixed_blend) 
    for u in guess_u_0 
    for t in guess_t_E
]


# Poszukiwanie najlepszego modelu
best_params, best_chi2 = find_best_model_parallel(initial_guesses, time, flux, flux_error)
fit_t_0, fit_u_0, fit_t_E, fit_source, fit_blending  = best_params
print(f"Najlepsze parametry: t_0 = {fit_t_0}, u_0 = {fit_u_0}, t_E = {fit_t_E}, Source: = {fit_source}, Blending: = {fit_blending} ")
print(f"Najlepsze chi-kwadrat: {best_chi2}")

# Obliczanie czasu dla zadanego wzmocnienia
def equation_for_time(A_0, t_E, u_0, t_0):
    a = A_0**2 - 1
    b = 4 * A_0**2 - 4
    c = -4
    delta = b**2 - 4 * a * c

    if delta <= 0:
        raise ValueError("Delta ujemna!")

    t1 = (-b - np.sqrt(delta)) / (2 * a)
    t2 = (-b + np.sqrt(delta)) / (2 * a)

    if t1 >= 0 and t2 >= 0:
        raise ValueError("t1 >= 0 i t2 >= 0 jednocześnie!")
    if t1 >= 0:
        u = np.sqrt(t1)
    if t2 >= 0:
        u = np.sqrt(t2)
    if (u**2 - u_0**2) < 0:
        raise ValueError("u**2 - u_0**2 < 0!")

    t_min = -np.sqrt(u**2 - u_0**2) * t_E + t_0
    t_max = np.sqrt(u**2 - u_0**2) * t_E + t_0

    return t_min, t_max

# Obliczanie modelu w pełnym zakresie czasów
time_model = np.linspace(np.min(time), np.max(time), 10000) 
u_model = np.sqrt(fit_u_0**2 + ((time_model - fit_t_0) / fit_t_E)**2)
A_model = ((u_model**2 + 2) / (u_model * np.sqrt(u_model**2 + 4)))
flux_model = fit_source * A_model+fit_blending

# Wizualizacja danych
#plt.scatter(time, magnitudo, label='Dane obserwacyjne', color='blue', s=10)
#plt.plot(time_model, -2.5 * np.log10(flux_model)+22, label='Model dopasowany', color='red', linewidth=2)
#plt.gca().invert_yaxis()  # Odwrócenie osi y, bo jasność rośnie w dół
#plt.xlabel('Czas')
#plt.ylabel('Jasność')
#plt.title('Dopasowanie modelu do danych fotometrycznych')
#plt.legend()
#plt.grid(True)
#plt.show()

#Usuwanie wzmocnienia
u_model = np.sqrt(fit_u_0**2 + ((time - fit_t_0) / fit_t_E)**2)
A_model = ((u_model**2 + 2) / (u_model * np.sqrt(u_model**2 + 4)))
flux_model_deleted =  (flux - fit_blending)/A_model 
new_magnuitudo = -2.5 * np.log10(flux_model_deleted)+22

# Wizualizacja danych
#plt.scatter(time, new_magnuitudo, label='Dane obserwacyjne', color='blue', s=10)
#plt.plot(time_model, -2.5 * np.log10(flux_model)+22, label='Model dopasowany', color='red', linewidth=2)
#plt.gca().invert_yaxis()  # Odwrócenie osi y, bo jasność rośnie w dół
#plt.xlabel('Czas')
#plt.ylabel('Jasność')
#plt.title('Dopasowanie modelu do danych fotometrycznych')
#plt.legend()
#plt.grid(True)
#plt.show()

# Pobranie bieżącej ścieżki folderu
current_path = os.getcwd()

# Nazwa bieżącego folderu
current_folder_name = os.path.basename(current_path)

# Pobranie ścieżki folderu nadrzędnego
parent_path = os.path.dirname(current_path)

# Nazwa folderu nadrzędnego
parent_folder_name = os.path.basename(parent_path)

in_filename = f"{parent_folder_name}_{current_folder_name}_in.dat"
out_filename = f"{parent_folder_name}_{current_folder_name}_out.dat"

# Wyświetlenie wyników
print("Nazwa pliku wejściowego:", in_filename)
print("Nazwa pliku wyjściowego:", out_filename)

#Zakłądam, że bierzemy miejsce gdzie 10% wynosi wzmocnienie
A_0_new =1+ sigma/fit_source
t1, t2 = equation_for_time(A_0_new, fit_t_E, fit_u_0, fit_t_0)
mask_in_range = (time >= t1) & (time <= t2)

# Maska dla danych wewnątrz zakresu
time_in_range = time[mask_in_range]
mag_deleted_in_range = new_magnuitudo[mask_in_range]
error_in_range = error[mask_in_range]
mag_in_range = magnitudo[mask_in_range]

# Maska dla danych poza zakresem
mask_out_of_range = ~mask_in_range
time_out_of_range = time[mask_out_of_range]
mag_deleted_out_of_range = new_magnuitudo[mask_out_of_range]
error_out_of_range = error[mask_out_of_range]
mag_out_of_range = magnitudo[mask_out_of_range]

if len(time_in_range) > 0:
    
    np.savetxt(in_filename, np.column_stack((time_in_range, mag_deleted_in_range, error_in_range)), fmt='%.8f %.8f %.8f', comments='')
    
    np.savetxt(out_filename, np.column_stack((time_out_of_range, mag_deleted_out_of_range, error_out_of_range)), fmt='%.8f %.8f %.8f', comments='')

else:
    print(f"Dane dla {in_filename} są puste, pomijam zapis.")
#plt.scatter(time_in_range, mag_in_range, label='Dane obserwacyjne I', color='blue', s=10)
#plt.scatter(time_out_of_range, mag_out_of_range, label='Dane obserwacyjne II', color='red', s=10)
#plt.gca().invert_yaxis()
#plt.xlabel('Czas')
#plt.ylabel('Jasność')
#plt.title('Dopasowanie modelu do danych fotometrycznych')
#plt.legend()
#plt.grid(True)
#plt.show()

# Wizualizacja danych
#plt.scatter(time_in_range, mag_deleted_in_range, label='Dane obserwacyjne I', color='blue', s=10)
#plt.scatter(time_out_of_range, mag_deleted_out_of_range, label='Dane obserwacyjne II', color='red', s=10)
#plt.gca().invert_yaxis()
#plt.xlabel('Czas')
#plt.ylabel('Jasność')
#plt.title('Dopasowanie modelu do danych fotometrycznych')
#plt.legend()
#plt.grid(True)
#plt.show()
