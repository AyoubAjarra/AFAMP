#importing libraries
import numpy as np


import pickle
from utils import *
import streaming_goldreich
import pandas as pd
from time import time
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
#Reproducibility
from utils import init_seed
init_seed(42)
n_iterations = 3
n = 43
teta = 1e-3

Sp_Summary = {}
def load_student_perf():
    X = np.load("student_perf_X.npy")
    y = np.load("student_perf_y.npy")
    a = np.load("student_perf_a.npy")
    return X, y, a
X, y , a= load_student_perf()
#a = np.where(a,1,-1)
X = np.interp(X, (X.min(), X.max()), (-1, +1))

h_star = LogisticRegression(penalty='l2', random_state=1, 
                              max_iter=5000).fit(X, y)

y = h_star.predict(X)
X_a1 = X[a == 1]
X_a0 = X[a == 0]
alpha = X_a1.shape[0]/(X_a1.shape[0] + X_a0.shape[0])
def evaluate_gf(h, X_a1, X_a0):
    pos_rate_a1 = np.average(h.predict(X_a1))
    pos_rate_a0 = np.average(h.predict(X_a0))
    return pos_rate_a1- pos_rate_a0
true_gf = abs(evaluate_gf(h_star, X_a1, X_a0))
Sp_Summary['true_sp'] = true_gf

Sp_Summary['non-efficient_afa'] = {}
Sp_Summary['efficient_afa'] = {}
Sp_Summary['iid'] = {}

#Baseline: uniform estimator
def iid_gf(label_budget, model = h_star):
    Sx1 = np.array([positive_random_input_generator(n) for _ in range(label_budget + 1)])
    Sx0 = np.array([negative_random_input_generator(n) for _ in range(label_budget + 1)])
    return abs(evaluate_gf(model, Sx1, Sx0))

IID_experiments = [dict([(k, []) for k in range(n_iterations)]) for _ in range(10)]  
computation_iid = []
for idx,experiment in enumerate(IID_experiments):
    for i in range(n_iterations):
        start_time_iid = time()
        experiment[i].append(iid_gf((idx+1)*100))
        end_time_iid = time()
        computation_iid.append(end_time_iid - start_time_iid)
Sp_Summary['iid']['computation_sp_iid'] = computation_iid


IID_df = pd.DataFrame(IID_experiments)
for i in range(n_iterations):
    for j in range(10):
        GF_column = "Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1) 
        IID_df[GF_column] = IID_df[i][j][0]
IID_df.drop([i for i in range(n_iterations)], axis = 1, inplace = True)

IID_table_GF = [[i+1,j+1,IID_df["Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1)][j]] for i in range(n_iterations) for j in range(10)]
IID_GF_experiments = pd.DataFrame(IID_table_GF, columns=['Iteration','Experiment','Estimated_Group_Fairness'])
IID_GF_experiments["error"] = abs(IID_GF_experiments["Estimated_Group_Fairness"] - true_gf)

Sp_Summary['iid']['experiments'] = IID_GF_experiments

## Efficient AFA

class Black_Box:
    def __init__(self, n):
        self.input_length = n
    def eval(self, x):
        return h_star.predict(x.reshape(1, -1))

# Find monomials
def find_monomials(label_budget, model = Black_Box(n), teta = teta ):
    gl = streaming_goldreich.Efficient_GoldreichLevin(model, teta, label_budget)
    return gl.find_heavy_monomials() # A list of tuples of subsets and their weights


def gf_estimator(label_budget, model = Black_Box(n)):
    S = [0 for _ in range(n)]
    sensitive_fourier = sum([monomial[1] for monomial in find_monomials( label_budget)])
    empty_fourier = estimate_fourier_coefficient(n,model, S, label_budget)
    a = 4* alpha * (1 - alpha)
    b = empty_fourier * (1 - 2* alpha )
    Delta  = empty_fourier**2 + 4 * alpha * ( alpha - 1) * (1+ 2* sensitive_fourier)
    if Delta < 0:
        return b /a
    return (b + Delta**.5)/a

computation_afa = []
AFA_experiments = [dict([(k, []) for k in range(n_iterations)]) for _ in range(10)]    

for idx,experiment in enumerate(AFA_experiments):
    for i in range(n_iterations):
        start_time = time()
        experiment[i].append(gf_estimator((idx+1)*100))
        end_time = time()
        computation_afa.append(end_time - start_time)
Sp_Summary['efficient_afa']['computation_time_eff_afa'] = computation_afa

AFA_efficient_df = pd.DataFrame(AFA_experiments)


for i in range(n_iterations):
    for j in range(10):
        GF_column = "Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1) 
        AFA_efficient_df[GF_column] = AFA_efficient_df[i][j][0]
AFA_efficient_df.drop([i for i in range(n_iterations)], axis = 1, inplace = True)

#Group Fairness experiments with AFA summary:
AFA_efficient_table_GF = [[i+1,j+1,AFA_efficient_df["Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1)][j]] for i in range(n_iterations) for j in range(10)]
AFA_efficient_GF_experiments = pd.DataFrame(AFA_efficient_table_GF, columns=['Iteration','Experiment','Estimated_Group_Fairness'])
AFA_efficient_GF_experiments["error"] = abs(AFA_efficient_GF_experiments["Estimated_Group_Fairness"] - true_gf)

Sp_Summary['efficient_afa']['experiments'] = AFA_efficient_GF_experiments

with open('student_SP_Experiments.pkl', 'wb') as handle:
    pickle.dump(Sp_Summary, handle)