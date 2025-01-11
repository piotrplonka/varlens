#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import math
import multiprocessing

#Wczytywanie nazwy pliku (głupi sposób XD)
def read_params_file(filename):
    field, star_no, ra, dec, remarks = None, None, None, None, None
    Tmax, Tmax_err = None, None
    tau, tau_err = None, None
    umin, umin_err = None, None
    Amax, Amax_err = None, None
    Dmag, Dmag_err = None, None
    fbl, fbl_err = None, None
    I_bl, I_bl_err = None, None
    I0, I0_err = None, None

    with open(filename, 'r') as file:
        lines = file.readlines()
        identifier = lines[0].strip()
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'Field':
                field = parts[1]
            elif parts[0] == 'StarNo':
                star_no = int(parts[1])
            elif parts[0] == 'RA(J2000.0)':
                ra = parts[1]
            elif parts[0] == 'Dec(J2000.0)':
                dec = parts[1]
            elif parts[0] == 'Remarks':
                remarks = " ".join(parts[1:]) if len(parts) > 1 else None
            elif parts[0] == 'Tmax':
                Tmax, Tmax_err = map(float, parts[1:3])
            elif parts[0] == 'tau':
                tau, tau_err = map(float, parts[1:3])
            elif parts[0] == 'umin':
                umin, umin_err = map(float, parts[1:3])
            elif parts[0] == 'Amax':
                Amax, Amax_err = map(float, parts[1:3])
            elif parts[0] == 'Dmag':
                Dmag, Dmag_err = map(float, parts[1:3])
            elif parts[0] == 'fbl':
                fbl, fbl_err = map(float, parts[1:3])
            elif parts[0] == 'I_bl':
                I_bl, I_bl_err = map(float, parts[1:3])
            elif parts[0] == 'I0':
                I0, I0_err = map(float, parts[1:3])

    # Przeniesienie wartości do osobnych zmiennych NumPy
    Tmax_np = np.array(Tmax)
    Tmax_err_np = np.array(Tmax_err)

    tau_np = np.array(tau)
    tau_err_np = np.array(tau_err)

    umin_np = np.array(umin)
    umin_err_np = np.array(umin_err)

    Amax_np = np.array(Amax)
    Amax_err_np = np.array(Amax_err)

    Dmag_np = np.array(Dmag)
    Dmag_err_np = np.array(Dmag_err)

    fbl_np = np.array(fbl)
    fbl_err_np = np.array(fbl_err)

    I_bl_np = np.array(I_bl)
    I_bl_err_np = np.array(I_bl_err)

    I0_np = np.array(I0)
    I0_err_np = np.array(I0_err)

    # Zwrócenie wyników
    return {
        "Identifier": identifier,
        "Field": field,
        "StarNo": star_no,
        "RA": ra,
        "Dec": dec,
        "Remarks": remarks,
        "Tmax": Tmax_np,
        "Tmax_err": Tmax_err_np,
        "tau": tau_np,
        "tau_err": tau_err_np,
        "umin": umin_np,
        "umin_err": umin_err_np,
        "Amax": Amax_np,
        "Amax_err": Amax_err_np,
        "Dmag": Dmag_np,
        "Dmag_err": Dmag_err_np,
        "fbl": fbl_np,
        "fbl_err": fbl_err_np,
        "I_bl": I_bl_np,
        "I_bl_err": I_bl_err_np,
        "I0": I0_np,
        "I0_err": I0_err_np
    }
    
filename = 'params.dat'
params = read_params_file(filename)

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
#print(f"Najlepsze parametry: t_0 = {fit_t_0}, u_0 = {fit_u_0}, t_E = {fit_t_E}, Baseline: = {fit_baseline}")
#print(f"Najlepsze chi-kwadrat: {best_chi2}")

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
#print(f"Najlepsze parametry po filtracji: t_0 = {fit_t_0}, u_0 = {fit_u_0}, t_E = {fit_t_E}")
#print(f"Najlepsze chi-kwadrat po filtracji: {best_chi2_filtered}")



#Dzielenie na dwa pliki
in_filename = f"{params['Identifier']}_in.dat"
out_filename = f"{params['Identifier']}_out.dat"

#Zakłądam, że bierzemy miejsce gdzie 1% wynosi wzmocnienie
A_0_new = 1.01
t1, t2 = equation_for_time(A_0_new, fit_t_E, fit_u_0, fit_t_0)
mask_in_range = (time >= t1) & (time <= t2)

# Maska dla danych wewnątrz zakresu
time_in_range = time[mask_in_range]
flux_in_range = flux[mask_in_range]
flux_error_in_range = flux_error[mask_in_range]

#Usuwanie
u_model_in_range = np.sqrt(fit_u_0**2 + ((time_in_range - fit_t_0) / fit_t_E)**2)
A_model_in_range = ((u_model_in_range**2 + 2) / (u_model_in_range * np.sqrt(u_model_in_range**2 + 4)))
flux_deleted_in_range = flux_in_range/A_model_in_range

mag_in_range = -2.5 * np.log10(flux_in_range)
mag_deleted_in_range = -2.5 * np.log10(flux_deleted_in_range)
mag_error_in_range =  -2.5 * np.log10(flux_error_in_range)

# Maska dla danych poza zakresem
mask_out_of_range = ~mask_in_range
time_out_of_range = time[mask_out_of_range]
flux_out_of_range = flux[mask_out_of_range]
flux_error_out_of_range = flux_error[mask_out_of_range]

#Usuwanie
u_model_out_of_range = np.sqrt(fit_u_0**2 + ((time_out_of_range - fit_t_0) / fit_t_E)**2)
A_model_out_of_range = ((u_model_out_of_range**2 + 2) / (u_model_out_of_range * np.sqrt(u_model_out_of_range**2 + 4)))
flux_deleted_out_of_range = flux_out_of_range/A_model_out_of_range

mag_out_of_range = -2.5 * np.log10(flux_deleted_out_of_range)
mag_deleted_out_of_range = -2.5 * np.log10(flux_out_of_range)
mag_error_out_of_range =  -2.5 * np.log10(flux_error_out_of_range)


# Wizualizacja danych
plt.scatter(time_in_range, mag_in_range, label='Dane obserwacyjne I', color='blue', s=10)
plt.scatter(time_out_of_range, mag_out_of_range, label='Dane obserwacyjne II', color='red', s=10)
plt.gca().invert_yaxis()  # Odwrócenie osi y, bo jasność rośnie w dół
plt.xlabel('Czas')
plt.ylabel('Jasność')
plt.title('Dopasowanie modelu do danych fotometrycznych')
plt.legend()
plt.grid(True)
plt.show()

# Wizualizacja danych
plt.scatter(time_in_range, mag_deleted_in_range, label='Dane obserwacyjne I', color='blue', s=10)
plt.scatter(time_out_of_range, mag_deleted_out_of_range, label='Dane obserwacyjne II', color='red', s=10)
plt.gca().invert_yaxis()  # Odwrócenie osi y, bo jasność rośnie w dół
plt.xlabel('Czas')
plt.ylabel('Jasność')
plt.title('Dopasowanie modelu do danych fotometrycznych')
plt.legend()
plt.grid(True)
plt.show()


# Zapisz do pliku z danymi "in"
np.savetxt(in_filename, np.column_stack((time_in_range, mag_deleted_in_range, mag_error_in_range)), fmt='%.8f %.8f %.8f', comments='')

# Zapisz do pliku z danymi "out"
np.savetxt(out_filename, np.column_stack((time_out_of_range, mag_deleted_out_of_range, mag_error_out_of_range)), fmt='%.8f %.8f %.8f', comments='')
