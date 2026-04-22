import numpy as np
import qibo
from rl_agent import Agent
from gym_env import QuantumCircuit
from utils import trace_distance, compute_fidelity, estimate_hardware_noise

qibo.set_backend("numpy")

estimate_dm_noise = True

shot = 50
exp_folder = f"shot/3q/{shot}/diagonal/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + f"dataset_{shot}shots.npz"
model_file_path = exp_folder + "model.zip"
eval_dataset_file = exp_folder + "eval_dataset.npz"
result_file_rl = exp_folder + "evaluation_result.npz"

env = QuantumCircuit(dataset_file=dataset_file, config_file=config_file)
agent = Agent(config_file=config_file, env=env, model_file_path=model_file_path)
result, dms_rl = agent.apply_eval_dataset(eval_dataset_file)

dataset = np.load(eval_dataset_file, allow_pickle=True)
dms_true = dataset['labels']

fidelities = []
trace_distances = []
mse_list = []
errors = []

for i in range(len(dms_true)):
    dm_rl, dm_true = dms_rl[i], dms_true[i]
    
    # Fidelity
    fidelities.append(compute_fidelity(dm_rl, dm_true))
    
    # Trace distance
    trace_distances.append(trace_distance(dm_rl, dm_true))
    
    # MSE
    mse_list.append(np.mean((dm_rl - dm_true)**2))
    
    # Hardware noise uncertainty
    if estimate_dm_noise:
        errors.append(estimate_hardware_noise(dms_rl=dm_rl, dms_true=dm_true))


fidelities = np.array(fidelities)
trace_distances = np.array(trace_distances)
mse_list = np.array(mse_list)
errors = np.array(errors)


avg_fid, std_fid = fidelities.mean(), fidelities.std()
avg_td, std_td = trace_distances.mean(), trace_distances.std()
avg_mse, std_mse = mse_list.mean(), mse_list.std()
avg_err = errors.mean() if estimate_dm_noise else None


print(f"Avg Fidelity: {avg_fid:.6f} ± {std_fid:.6f}")
print(f"Avg MSE: {avg_mse:.12f} ± {std_mse:.12f}")
print(f"Avg Trace Distance: {avg_td:.6f} ± {std_td:.6f}")
if estimate_dm_noise:
    print(f"Avg Measurement Uncertainty (Error): {avg_err:.6f}")

# Save results
np.savez(result_file_rl, result=result, dms=dms_rl)
