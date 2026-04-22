from rl_agent import Agent
from gym_env import QuantumCircuit
import qibo
import numpy as np
import json
qibo.set_backend("numpy")


shot = 50
exp_folder = f"shot/3q/{shot}/diagonal/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + f"dataset_{shot}shots.npz"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

'''config = json.load(open(config_file))
use_diagonal = config["dataset"]["only_diagonal"]

data = np.load(dataset_file)
y = data["labels"]

if use_diagonal:
    y = np.array([np.diag(dm) for dm in y])

idx = np.random.randint(len(y))   # indice casuale
print(f"Diagonale #{idx}:")
print(y[idx])  
'''

agent = Agent(config_file = config_file, env = env)
agent.train(n_steps = 300_000)