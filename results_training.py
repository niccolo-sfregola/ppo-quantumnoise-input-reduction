import numpy as np

shot = 50

path = f"shot/3q/{shot}/diagonal/"
results = path + "model_train_result.npz"

data = np.load(results, allow_pickle=True)
print(data.keys())
timesteps = data["timesteps"]
train_results = data["train_results"].squeeze()
val_results = data["val_results"].squeeze()


best_idx = np.argmax(val_results["fidelity"])

best_fid = train_results["fidelity"][best_idx]
best_fid_std = train_results["fidelity_std"][best_idx]
best_td = train_results["trace_distance"][best_idx]
best_td_std = train_results["trace_distance_std"][best_idx]
best_timestep = timesteps[best_idx]

print("--------------TRAINING!!!!---------------")
print(f"Best model salvato @ k={best_timestep:.0f}k steps")
print(f"  Fidelity: {best_fid:.4f} ± {best_fid_std:.4f}")
print(f"  Trace Distance: {best_td:.4f} ± {best_td_std:.4f}")

best_fid = val_results["fidelity"][best_idx]
best_fid_std = val_results["fidelity_std"][best_idx]
best_td = val_results["trace_distance"][best_idx]
best_td_std = val_results["trace_distance_std"][best_idx]
best_timestep = timesteps[best_idx]


print("--------------VALIDATION!!!!---------------")

print(f"Best model salvato @ k={best_timestep:.0f}k steps")
print(f"  Fidelity: {best_fid:.4f} ± {best_fid_std:.4f}")
print(f"  Trace Distance: {best_td:.4f} ± {best_td_std:.4f}")
