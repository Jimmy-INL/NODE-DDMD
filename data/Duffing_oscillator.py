import warnings

warnings.filterwarnings('ignore')

from Duffing_oscillator_ODE import Duffing_oscillator_ODE
import csv
import numpy as np

numICs = 10000
filenamePrefix = 'Duffing_oscillator'

x1range = [-2, 2]
x2range = [-2, 2]
tSpan = np.arange(0, 2.5 + 0.1, 0.25)# np.arange(0, 125, 0.25)  # 0:0.02:1



def make_csv(filename, X):
    with open(filename, 'w') as csv_file:
        fieldnames = ['precision_x1', 'precision_x2']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for i in range(len(X)):
            writer.writerow({'precision_x1': X[i, 0], 'precision_x2': X[i, 1]})

seed = 1
X_test = Duffing_oscillator_ODE(x1range, x2range, round(1 * numICs), tSpan, seed)
filename_test = filenamePrefix + '_test_x.csv'
make_csv(filename_test, X_test)

seed = 2
X_val = Duffing_oscillator_ODE(x1range, x2range, round(1 * numICs), tSpan, seed)
filename_val = filenamePrefix + '_val_x.csv'
make_csv(filename_val, X_val)

seed = 100
X_train = Duffing_oscillator_ODE(x1range, x2range, round(1 * numICs), tSpan, seed)
filename_train = filenamePrefix + '_train_x.csv'
make_csv(filename_train, X_train)

"""for j in range(1, 7):
    seed = 2 + j
    X_train = Duffing_oscillator_ODE(x1range, x2range, round(.7 * numICs), tSpan, seed)
    filename_train = filenamePrefix + format('_train%d_x.csv' % j)
    make_csv(filename_train, X_train)"""