import warnings

warnings.filterwarnings('ignore')

from Duffing_oscillator_ODE import Duffing_oscillator_ODE
from Discrete_Linear_ODE import Discrete_Linear_ODE
from Linear_ODE import Linear_ODE
import csv
import numpy as np

numICs = 1000

x1range = [-2, 2]
x2range = [-2, 2]
tSpan = np.arange(0, 2.5 + 0.1, 0.25)# np.arange(0, 125, 0.25)  # 0:0.02:1



def make_csv(filename, X):
    with open(filename, 'w') as csv_file:
        fieldnames = ['precision_x1', 'precision_x2']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for i in range(len(X)):
            writer.writerow({'precision_x1': X[i, 0], 'precision_x2': X[i, 1]})





###############Discrete_Linear#############################
filenamePrefix = 'spectrum'
seed = 10
X_train = Discrete_Linear_ODE(x1range, x2range, 100, np.arange(0, 0.25 - 0.1, 0.25), seed, "x") # 0, 2.5, 0.25
filename_train = filenamePrefix + '_train_x.csv'
make_csv(filename_train, X_train)

seed = 10
tSpan = np.arange(0, 0.25 + 0.1, 0.25) # 0, 2.5 + 0.1, 0.25
X_train = Discrete_Linear_ODE(x1range, x2range, 100, tSpan, seed, "y")
filename_train = filenamePrefix + '_train_y.csv'
make_csv(filename_train, X_train)

seed = 1
tSpan = np.arange(0, 12.5, 0.25)  # 0, 12.5, 0.25
X_train = Discrete_Linear_ODE(x1range, x2range, round(1 * numICs), tSpan, seed)
filename_train = filenamePrefix + "_E_recon_50" + '.csv'
make_csv(filename_train, X_train)

seed = 3
tSpan = np.arange(0, 0.25 + 0.1, 0.25)  # 0, 12.5, 0.25
X_train = Linear_ODE(x1range, x2range, 10000, tSpan, seed)
filename_train = filenamePrefix + "_E_eigfunc" + '.csv'
make_csv(filename_train, X_train)
"""seed = 1
X_test = Duffing_oscillator_ODE(x1range, x2range, round(1 * numICs), tSpan, seed)
filename_test = filenamePrefix + '_test_x.csv'
make_csv(filename_test, X_test)

seed = 2
X_val = Duffing_oscillator_ODE(x1range, x2range, round(1 * numICs), tSpan, seed)
filename_val = filenamePrefix + '_val_x.csv'
make_csv(filename_val, X_val)

seed = 100
X_train = Duffing_oscillator_ODE(x1range, x2range, 100000, tSpan, seed)
filename_train = filenamePrefix + '_train_x.csv'
make_csv(filename_train, X_train)
"""