#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
import MulensModel as mm
import scipy.optimize as op
import multiprocessing


filename_data = 'phot.dat'
data = np.loadtxt(filename_data, usecols=(0, 1, 2))
time, magnitudo, error = data[:, 0], data[:, 1], data[:, 2]




sorted_indices = np.argsort(magnitudo)
lowest_magnitudes_indices = sorted_indices[:10]
lowest_magnitudes_time = time[lowest_magnitudes_indices]
median_time = np.median(lowest_magnitudes_time)




Data_Lens = mm.MulensData(
    data_list=(time, magnitudo, error),
    phot_fmt='mag',
    add_2450000=False,
)




def chi2_fun(theta, parameters_to_fit, event):
    for (parameter, value) in zip(parameters_to_fit, theta):
        if parameter == 't_E' and value < 0:  
            return np.inf  
        setattr(event.model.parameters, parameter, value)
    return event.get_chi2()




def process_model(guess):
    params = {'t_0': median_time, 'u_0': guess[0], 't_E': guess[1]}
    if params['t_E'] < 0:  # Pomijanie przypadków z ujemnym t_E
        return None
    model = mm.Model(params)
    event = mm.Event(datasets=(Data_Lens,), model=model)

    result = op.minimize(
        chi2_fun, x0=[params['t_0'], params['u_0'], params['t_E']],
        args=(['t_0', 'u_0', 't_E'], event),
        method='Nelder-Mead'
    )
    chi2 = chi2_fun(result.x, ['t_0', 'u_0', 't_E'], event)
    return (result.x, model, event, chi2)




def find_best_model_parallel(initial_guesses, num_processes=12):
    best_result = None
    best_chi2 = np.inf
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_model, initial_guesses)
    for result in results:
        if result is not None:
            _, model, event, chi2 = result
            print(f"Chi-kwadrat: {chi2}")
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_result = result
    return best_result


guess_u_0 = np.linspace(0.01, 1, 10)  
guess_t_E = np.linspace(5, 200, 10)     
initial_guesses = [(u, t) for u in guess_u_0 for t in guess_t_E]

best_params, best_model, best_event, best_chi2 = find_best_model_parallel(initial_guesses)
fit_t_0, fit_u_0, fit_t_E = best_params

print(f"Najlepsze parametry: t_0 = {fit_t_0}, u_0 = {fit_u_0}, t_E = {fit_t_E}")
print(f"Najlepsze chi-kwadrat: {best_chi2}")


plt.figure()
best_event.plot_data()
tmin, tmax = np.min(time), np.max(time)
best_event.plot_model(t_range=[tmin, tmax], color='red')
plt.grid()
plt.title('Dane i najlepszy dopasowany model')
plt.show()

baseline = np.median(magnitudo)
print(baseline)

def delete_ampli(mag, t, t_0, u_0, t_E):
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return mag + 2.5*np.log10(A)

new_magnitude = delete_ampli(magnitudo, time, fit_t_0, fit_u_0, fit_t_E)


plt.figure()
plt.scatter(time, new_magnitude, label='Skorygowana jasność')
plt.gca().invert_yaxis()
best_event.plot_model(t_range=[tmin, tmax], color='red', label='Model')
plt.xlabel('Czas')
plt.ylabel('Jasność')
plt.grid(True)
plt.legend()
plt.show()

