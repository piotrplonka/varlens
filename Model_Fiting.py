#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import math
import multiprocessing

# Wczytywanie danych z pliku z fotometrią
filename_data = 'phot.dat'
data = np.loadtxt(filename_data, usecols=(0, 1, 2))
time, magnitudo, error = data[:, 0], data[:, 1], data[:, 2]
flux = 10**(-0.4 * magnitudo)
flux_error = 10**(-0.4 * error)

# Poszukiwanie maksimum jasności jako mediany 10 najjaśniejszych punktów
sorted_indices = np.argsort(magnitudo)
lowest_magnitudes_indices = sorted_indices[:10]
lowest_magnitudes_time = time[lowest_magnitudes_indices]
median_time = np.median(lowest_magnitudes_time)

#Baseline początkowa wartość
baseline_guess_flux=np.median(flux)
print(baseline_guess_flux)

# Funkcja chi-kwadrat 
def chi2_fun(params, time, flux, flux_error):
    t_0, u_0, t_E, baseline = params
    if t_E < 0:
        return np.inf
    u = np.sqrt(u_0**2 + ((time - t_0) / t_E)**2)
    A = ((u**2 + 2) / (u * np.sqrt(u**2 + 4)))
    flux_calculated = baseline * A
    chi2 = np.sum(((flux - flux_calculated) / flux_error)**2)
    return chi2

# Proces dopasowania modelu 
def process_model(guess, time, flux, flux_error):
    result = op.minimize(
        chi2_fun, x0=guess, args=(time, flux, flux_error), method='Nelder-Mead', 
        options={
            'xatol': 1e-8,   # Dokładność w parametrach
            'fatol': 1e-8,   # Dokładność w funkcji celu
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
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_result = params

    return best_result, best_chi2

# Tablice wartości początkowych
guess_u_0 = np.linspace(0.01, 1, 10)
guess_t_E = np.linspace(1, 200, 10)
baseline_guess_flux = np.median(flux)  

# Dodanie baseline do wartości początkowych
initial_guesses = [(median_time, u, t, baseline_guess_flux) for u in guess_u_0 for t in guess_t_E]

# Poszukiwanie najlepszego modelu
best_params, best_chi2 = find_best_model_parallel(initial_guesses, time, flux, flux_error)
fit_t_0, fit_u_0, fit_t_E, fit_baseline = best_params
print(f"Najlepsze parametry: t_0 = {fit_t_0}, u_0 = {fit_u_0}, t_E = {fit_t_E}, Baseline: = {fit_baseline}")
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

# Filtracja danych
A_0 = 1.001
t1, t2 = equation_for_time(A_0, fit_t_E, fit_u_0, fit_t_0)
mask = (time >= t1) & (time <= t2)
filtered_time = time[mask]
filtered_flux = flux[mask]
filtered_flux_error = flux_error[mask]

# Dopasowanie modelu do przefiltrowanych danych
best_params_filtered, best_chi2_filtered = process_model(
    best_params, filtered_time, filtered_flux, filtered_flux_error
)
fit_t_0, fit_u_0, fit_t_E, fit_baseline = best_params_filtered
print(f"Najlepsze parametry po filtracji: t_0 = {fit_t_0}, u_0 = {fit_u_0}, t_E = {fit_t_E}")
print(f"Najlepsze chi-kwadrat po filtracji: {best_chi2_filtered}")

# Obliczanie modelu w pełnym zakresie czasów
time_model = np.linspace(np.min(time), np.max(time), 10000) 
u_model = np.sqrt(fit_u_0**2 + ((time_model - fit_t_0) / fit_t_E)**2)
A_model = ((u_model**2 + 2) / (u_model * np.sqrt(u_model**2 + 4)))
flux_model = fit_baseline * A_model

# Wizualizacja danych
plt.scatter(time, magnitudo, label='Dane obserwacyjne', color='blue', s=10)
plt.plot(time_model, -2.5 * np.log10(flux_model), label='Model dopasowany', color='red', linewidth=2)
plt.gca().invert_yaxis()  # Odwrócenie osi y, bo jasność rośnie w dół
plt.xlabel('Czas')
plt.ylabel('Jasność')
plt.title('Dopasowanie modelu do danych fotometrycznych')
plt.legend()
plt.grid(True)
plt.show()




#
u_model = np.sqrt(fit_u_0**2 + ((time - fit_t_0) / fit_t_E)**2)
A_model = ((u_model**2 + 2) / (u_model * np.sqrt(u_model**2 + 4)))
flux_model_deleted = flux / A_model
new_magnuitudo = -2.5 * np.log10(flux_model_deleted)






# Wizualizacja danych
plt.scatter(time, new_magnuitudo, label='Dane obserwacyjne', color='blue', s=10)
plt.plot(time_model, -2.5 * np.log10(flux_model), label='Model dopasowany', color='red', linewidth=2)
plt.gca().invert_yaxis()  # Odwrócenie osi y, bo jasność rośnie w dół
plt.xlabel('Czas')
plt.ylabel('Jasność')
plt.title('Dopasowanie modelu do danych fotometrycznych')
plt.legend()
plt.grid(True)
plt.show()
