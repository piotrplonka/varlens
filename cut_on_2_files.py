#!/usr/bin/python3
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
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
#print("Identifier:", params["Identifier"])
#print("Field:", params["Field"])
#print("StarNo:", params["StarNo"])
#print("RA:", params["RA"])
#print("Dec:", params["Dec"])
#print("Remarks:", params["Remarks"])
#print("Tmax:", params["Tmax"], "±", params["Tmax_err"])
#print("tau:", params["tau"], "±", params["tau_err"])
#print("umin:", params["umin"], "±", params["umin_err"])
#print("Amax:", params["Amax"], "±", params["Amax_err"])
#print("Dmag:", params["Dmag"], "±", params["Dmag_err"])
#print("fbl:", params["fbl"], "±", params["fbl_err"])
#print("I_bl:", params["I_bl"], "±", params["I_bl_err"])
#print("I0:", params["I0"], "±", params["I0_err"])

filename = 'params.dat'
params = read_params_file(filename)


filename_data = 'phot.dat'
data = np.loadtxt(filename_data)
time, magnitudo, error = data[:, 0], data[:, 1], data[:, 2]


newtime = np.linspace(np.min(time), np.max(time), 1000)
u = np.sqrt(params["umin"]**2 + ((newtime - params["Tmax"]) / params["tau"])**2)
A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))  

magnitudo_theoretical = -2.5 * np.log10(A) + params["I_bl"]

def delete_ampli(mag,t):
    u = np.sqrt(params["umin"]**2 + ((t - params["Tmax"]) / params["tau"])**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    ampli=-2.5 * np.log10(A) #+ params["I_bl"]
    return  mag+2.5 * np.log10(A)

new_magnitude=delete_ampli(magnitudo,time)

def A_function(t, params):
    u = np.sqrt(params["umin"]**2 + ((t - params["Tmax"]) / params["tau"])**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A
    
def find_t(amp, params):

    def equation(t):
        return A_function(t, params) - amp
    t0 = params["Tmax"]  +5
    t_solution = fsolve(equation, t0)
    
    return t_solution[0]
#print((100**(1/5))**np.std(magnitudo))
amp =1+0.02*(100**(1/5))**np.std(magnitudo)#1.05#*(100**(1/5))**np.std(magnitudo)
#print(amp)
t_result = find_t(amp, params)
#print(f"Znaleziony czas t, dla którego A(t) = {amp}: {t_result}")
#print(abs(params["Tmax"]-t_result))

t_min = params["Tmax"] - abs(params["Tmax"] - t_result)
t_max = params["Tmax"] + abs(params["Tmax"] - t_result)

mask_in_range = (time >= t_min) & (time <= t_max)
mask_out_of_range = ~mask_in_range
data_in_range = data[mask_in_range]
data_out_of_range = data[mask_out_of_range]


time_in, magnitudo_in, error_in = data_in_range[:, 0], data_in_range[:, 1], data_in_range[:, 2]
time_out, magnitudo_out, error_out = data_out_of_range[:, 0], data_out_of_range[:, 1], data_out_of_range[:, 2]

newmag_in=delete_ampli(magnitudo_in,time_in)
newmag_out=delete_ampli(magnitudo_out,time_out)

in_filename = f"{params['Identifier']}_in.dat"
out_filename = f"{params['Identifier']}_out.dat"

# Zapisz do pliku z danymi "in"
np.savetxt(in_filename, np.column_stack((time_in, newmag_in, error_in)), fmt='%.8f %.8f %.8f', comments='')

# Zapisz do pliku z danymi "out"
np.savetxt(out_filename, np.column_stack((time_out, newmag_out, error_out)), fmt='%.8f %.8f %.8f', comments='')

print(f"Dane zapisane do plików: {in_filename}, {out_filename}")

#plt.figure(figsize=(10, 6))
#plt.errorbar(time_in, magnitudo_in, yerr=error_in, fmt='o', markersize=4, capsize=3, label='Obserwacje', color='blue')
#plt.errorbar(time_out, magnitudo_out, yerr=error_out, fmt='o', markersize=4, capsize=3, label='Obserwacje', color='red')
##plt.plot(newtime, magnitudo_theoretical, '-', color='red', linewidth=2, label='Krzywa teoretyczna')
#plt.gca().invert_yaxis()
#plt.xlabel("Czas (time)", fontsize=12)
#plt.ylabel("Magnitudo", fontsize=12)
#plt.title("Wykres Time vs Magnitudo", fontsize=14)
#plt.legend()
#plt.grid()
#plt.show()

#plt.figure(figsize=(10, 6))
#plt.errorbar(time_out, magnitudo_out, yerr=error_out, fmt='o', markersize=4, capsize=3, label='Obserwacje', color='blue')
#plt.plot(newtime, magnitudo_theoretical, '-', color='red', linewidth=2, label='Krzywa teoretyczna')
#plt.gca().invert_yaxis()
#plt.xlabel("Czas (time)", fontsize=12)
#plt.ylabel("Magnitudo", fontsize=12)
#plt.title("Wykres Time vs Magnitudo", fontsize=14)
#plt.legend()
#plt.grid()
#plt.show()

#plt.figure(figsize=(10, 6))
#plt.errorbar(time_in, magnitudo_in, yerr=error_in, fmt='o', markersize=4, capsize=3, label='Obserwacje', color='blue')
#plt.plot(newtime, magnitudo_theoretical, '-', color='red', linewidth=2, label='Krzywa teoretyczna')
#plt.gca().invert_yaxis()
#plt.xlabel("Czas (time)", fontsize=12)
#plt.ylabel("Magnitudo", fontsize=12)
#plt.title("Wykres Time vs Magnitudo", fontsize=14)
#plt.legend()
#plt.grid()
#plt.show()
