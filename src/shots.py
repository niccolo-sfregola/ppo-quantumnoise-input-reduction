from qibo import set_backend
from utils_hardware import state_tomography
from circuit_representation import CircuitRepresentation
from noise_model import CustomNoiseModel
from qibo.backends import NumpyBackend
import numpy as np
from qibo.config import log
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit


log.setLevel("WARNING")   

backend = NumpyBackend()





exp_folder = "shot/3q/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"


def fidelity_model(N, Finf, A):
    return Finf - A / np.sqrt(N)

def td_model(N, Dinf, B):
    return Dinf + B / np.sqrt(N)




def labels_from_shots(dataset_file,
                       nshots, 
                       config_file,
                       output_file):
    
    """Builds density-matrices using a finite number of shots"""
    set_backend("numpy")
    circuits_rep = np.load(dataset_file, allow_pickle=True)['circuits']
    rep = CircuitRepresentation(config_file)
    noise_model = CustomNoiseModel(config_file) 

    labels = []

    for c in circuits_rep:
        circ = rep.rep_to_circuit(c)
        if noise_model:
            circ = noise_model.apply(circ)
        rho = state_tomography([circ], nshots, likelihood= True, backend=backend)[0][2]
        labels.append(rho)
    
    
    labels = np.stack(labels).astype(np.complex128)
    np.savez(output_file, circuits=circuits_rep, labels=labels)


    return output_file


def shot_plot(config_file, shots, fidelities, std_fidelities, tds, std_tds, fidelities2, std_fidelities2, tds2, std_tds2, folder,diagonal:bool,fit:bool):
    with open(config_file, "r") as f:
        config = json.load(f)

    qubits     = config["dataset"]["qubits"]
    lambda_val = config["noise"]["dep_lambda"]
    p0         = config["noise"]["p0"]
    eps_x      = config["noise"]["epsilon_x"]
    eps_z      = config["noise"]["epsilon_z"]
    SMALL_SIZE = 22
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 28

    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    


    shots = np.asarray(shots, dtype=float)

    if fit:

        # Fit Fidelity
        if not diagonal:
            pF, covF = curve_fit(fidelity_model, shots, fidelities,
                                p0=[fidelities[-1], (1 - fidelities[-1]) * np.sqrt(shots[-1])])
            Finf, A = pF
            sigma_Finf, sigma_A = np.sqrt(np.diag(covF))

            pD, covD = curve_fit(td_model, shots, tds,
                            p0=[tds[-1], (tds[0] - tds[-1]) * np.sqrt(shots[0])])
            Dinf, B = pD
            sigma_Dinf, sigma_B = np.sqrt(np.diag(covD))
            Nfit = np.geomspace(min(shots), max(shots), 200)
            F_fit = fidelity_model(Nfit, *pF)
            D_fit = td_model(Nfit, *pD)

            sigma_F = np.sqrt(sigma_Finf**2 + (sigma_A / np.sqrt(Nfit))**2)
            sigma_D = np.sqrt(sigma_Dinf**2 + (sigma_B / np.sqrt(Nfit))**2)

        else:

            pF2, covF2 = curve_fit(fidelity_model, shots, fidelities2,
                                p0=[fidelities2[-1], (1 - fidelities2[-1]) * np.sqrt(shots[-1])])
            Finf2, A2 = pF2
            sigma_Finf2, sigma_A2 = np.sqrt(np.diag(covF2))
            pD2, covD2 = curve_fit(td_model, shots, tds2,
                            p0=[tds2[-1], (tds2[0] - tds2[-1]) * np.sqrt(shots[0])])
            Dinf2, B2 = pD2
            sigma_Dinf2, sigma_B2 = np.sqrt(np.diag(covD2))
            Nfit = np.geomspace(min(shots), max(shots), 200)
            F_fit2 = fidelity_model(Nfit, *pF2)
            D_fit2 = td_model(Nfit, *pD2)

            sigma_F2 = np.sqrt(sigma_Finf2**2 + (sigma_A2 / np.sqrt(Nfit))**2)
            sigma_D2 = np.sqrt(sigma_Dinf2**2 + (sigma_B2 / np.sqrt(Nfit))**2)





    fig, ax = plt.subplots(1, 2, figsize=(24, 9))

    # --- Fidelity ---

    if not diagonal:
        ax[0].errorbar(shots, fidelities, yerr=std_fidelities, fmt='o', capsize=2, color='darkred', label='Data - Full representation')
        if fit:
            ax[0].plot(Nfit, F_fit, '--', color='red',
               label=rf"Fit full ±1$\sigma$")
            ax[0].fill_between(Nfit, F_fit - sigma_F, F_fit + sigma_F,
                       alpha=0.2, color='red')
        ax[0].set_xscale('log')
        ax[0].set(xlabel=r'$n^o$ of shots', ylabel='Fidelity')
        ax[0].set_title("Fidelity vs number of shots")
        ax[0].grid(linestyle='--', alpha=0.6)
        ax[0].legend(fontsize=15)

        ax[1].errorbar(shots, tds, yerr=std_tds, fmt='o', capsize=2, color='darkred', label='Data - Full representation')
        if fit:
            ax[1].plot(Nfit, D_fit, '--', color='red',
                label=rf"Fit full ±1$\sigma$")
            ax[1].fill_between(Nfit, D_fit - sigma_D, D_fit + sigma_D,
                        alpha=0.2, color='red')
        ax[1].set_xscale('log')
        ax[1].set(xlabel=r'$n^o$ of shots', ylabel='Trace Distance')
        ax[1].set_title("Trace Distance vs number of shots")
        ax[1].grid(linestyle='--', alpha=0.6)
        ax[1].legend(fontsize=15)
        suptitle = (
        fr"$n^o$ of qubits: {qubits} | "
        fr"Noise Params: $\lambda$={lambda_val}, $\gamma$={p0}, "
        fr"$\theta_x$={eps_x}, $\theta_z$={eps_z}"
        )
        plt.suptitle(suptitle, fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(folder + f"images/shot_results_{qubits}q_full.pdf")
        plt.show()

        print(f"F_inf = {Finf:.3f} pm {sigma_Finf:.3f}\n A = {A:.3f} pm {sigma_A:.3f}\n TD_inf = {Dinf:.3f} pm {sigma_Dinf:.3f}\n B = {B:.3f} pm {sigma_B:.3f}")
    else:
        ax[0].errorbar(shots, fidelities2, yerr=std_fidelities2, fmt='o', capsize=2, color='darkblue', label='Data - Diagonal representation')

        if fit:

            ax[0].plot(Nfit, F_fit2, '--', color='blue',
                label=rf"Fit diagonal ±1$\sigma$")
            ax[0].fill_between(Nfit, F_fit2 - sigma_F2, F_fit2 + sigma_F2,
                        alpha=0.2, color='blue')
        ax[0].set_xscale('log')
        ax[0].set(xlabel=r'$n^o$ of shots', ylabel='Fidelity')
        ax[0].set_title("Fidelity vs number of shots")
        ax[0].grid(linestyle='--', alpha=0.6)
        ax[0].legend(fontsize=15)

        # --- Trace Distance ---

        ax[1].errorbar(shots, tds2, yerr=std_tds2, fmt='o', capsize=2, color='darkblue', label='Data - Diagonal representation')
        if fit:

            ax[1].plot(Nfit, D_fit2, '--', color='blue',
                label=rf"Fit diagonal ±1$\sigma$")
            ax[1].fill_between(Nfit, D_fit2 - sigma_D2, D_fit2 + sigma_D2,
                        alpha=0.2, color='blue')
        ax[1].set_xscale('log')
        ax[1].set(xlabel=r'$n^o$ of shots', ylabel='Trace Distance')
        ax[1].set_title("Trace Distance vs number of shots")
        ax[1].grid(linestyle='--', alpha=0.6)
        ax[1].legend(fontsize=15)

        suptitle = (
            fr"$n^o$ of qubits: {qubits} | "
            fr"Noise Params: $\lambda$={lambda_val}, $\gamma$={p0}, "
            fr"$\theta_x$={eps_x}, $\theta_z$={eps_z}"
        )
        plt.suptitle(suptitle, fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(folder + f"images/shot_results_{qubits}q_diagonal.pdf")
        plt.show()
        print(f"F_inf = {Finf2:.3f} pm {sigma_Finf2:.3f}\n A = {A2:.3f} pm {sigma_A2:.3f}\n TD_inf = {Dinf2:.3f} pm {sigma_Dinf2:.3f}\n B = {B2:.3f} pm {sigma_B2:.3f}")





    


'''shots = [3]
for s in shots:
    shot_file = f"{exp_folder}dataset_{s}shots.npz"
    labels_from_shots(dataset_file,s,config_file,shot_file)
'''
