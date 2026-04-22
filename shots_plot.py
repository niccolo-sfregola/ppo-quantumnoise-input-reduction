from shots import shot_plot
import numpy as np

folder = "shot/1q/"
config_file = folder + "config.json"

shots = np.array([3,10,50,200,1000])


fidelities = np.array([0.856107, 0.931447, 0.970744, 0.989530, 0.987703])
std_f = np.array([ 0.028589, 0.023758, 0.010433, 0.011936, 0.012569])

tds = np.array([ 0.200778, 0.161459, 0.110713, 0.086252, 0.064726])
std_t = np.array([ 0.045984, 0.049143,0.042736, 0.045987, 0.044980])



fidelities2 = np.array([0.895683, 0.956825, 0.987513, 0.987481,  0.989594 ])  
std_f2 = np.array([0.029000, 0.015777, 0.016230, 0.016822, 0.012371])

tds2 = np.array([0.184462, 0.097079, 0.088836, 0.090066, 0.087972]) 
std_t2 = np.array([0.047425, 0.050631, 0.052248, 0.053053, 0.046864])






shot_plot(config_file, shots, fidelities, std_f, tds, std_t,fidelities2, std_f2, tds2, std_t2, folder,diagonal = False, fit =True)