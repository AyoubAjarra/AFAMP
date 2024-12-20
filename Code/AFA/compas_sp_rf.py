#importing libraries
import numpy as np
from sklearn import cluster, datasets
from sklearn import svm
from learning import *
import cvxpy as cp
from numpy.linalg import norm

import pickle
from utils import *
import streaming_goldreich
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
#Reproducibility
from utils import init_seed
init_seed(42)
n_iterations = 10
n = 13
teta = 1e-3

Sp_Summary = {}
df = pd.read_csv("compas_arrest_processed.csv")
#df["arrest"]= np.where(df["arrest"] == -1, 0, df["arrest"])
for column in df.columns[1:]:
    df[column] = (df[column]).astype(int)
    df[column]= np.where(df[column] == 0, -1, df[column])

columns_delete = [ 'n_juvenile_misdemeanors_eq_0', 'n_juvenile_misdemeanors_geq_1', 
                  'n_juvenile_misdemeanors_geq_2', 'n_juvenile_misdemeanors_geq_5', 'n_juvenile_felonies_eq_0', 
                  'n_juvenile_felonies_geq_1', 'n_juvenile_felonies_geq_2', 'n_juvenile_felonies_geq_5']
df = df.drop(columns_delete, axis=1)

def load_compas():
    y = np.array(df["arrest"] == 1, dtype=int)   
    a = np.array(df["race_is_causasian"])
    X = df.to_numpy().astype('int')[:, 1:] #get rid off label
    return X, y, a

X, y, a= load_compas()

h_star = RandomForestClassifier(n_estimators = 100).fit(X, y)

y = h_star.predict(X)
X_a1 = X[a == 1]
X_a0 = X[a == -1]
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
Sp_Summary['cal_baselines'] = {}
#Baseline: uniform estimator
def iid_gf(label_budget, model = h_star):
    Sx1 = np.array([positive_random_input_generator(n) for _ in range(label_budget + 1)])
    Sx0 = np.array([negative_random_input_generator(n) for _ in range(label_budget + 1)])
    return abs(evaluate_gf(model, Sx1, Sx0))
#Ours
class Black_Box:
    def __init__(self, n):
        self.input_length = n
    def eval(self, x):
        return h_star.predict(x.reshape(1, -1))

# Find monomials
def find_monomials(label_budget, model = Black_Box(n), teta = teta ):
    gl = streaming_goldreich.GoldreichLevin(model, teta, label_budget)
    return gl.find_heavy_monomials() # A list of tuples of subsets and their weights


def gf_estimator(label_budget, model = Black_Box(n)):
    S = [0 for _ in range(n)]
    monomials = find_monomials( label_budget)
    sensitive_fourier = sum([estimate_fourier_coefficient(n,model, S_a, label_budget)**2 for S_a in generate_lists(n)])
    empty_fourier = estimate_fourier_coefficient(n,model, S, label_budget)
    a = 4* alpha * (1 - alpha)
    b = empty_fourier * (1 - 2* alpha )
    Delta  = empty_fourier**2 + 4 * alpha * ( alpha - 1) * (1+ 2* sensitive_fourier)
    if Delta < 0:
        return b /a
    return (b + Delta**.5)/a

start_time_afa = time()
AFA_experiments = [dict([(k, []) for k in range(n_iterations)]) for _ in range(10)]    
for idx,experiment in enumerate(AFA_experiments):
    for i in range(n_iterations):
        experiment[i].append(gf_estimator((idx+1)*100))
end_time_afa = time()
computation_sp_afa = end_time_afa - start_time_afa
Sp_Summary['non-efficient_afa']['computation_sp_afa'] = computation_sp_afa


start_time_iid = time()
IID_experiments = [dict([(k, []) for k in range(n_iterations)]) for _ in range(10)]  
for idx,experiment in enumerate(IID_experiments):
    for i in range(n_iterations):
        experiment[i].append(iid_gf((idx+1)*100))
end_time_iid = time()
computation_sp_iid = end_time_iid - start_time_iid
Sp_Summary['iid']['computation_sp_iid'] = computation_sp_iid

AFA_df = pd.DataFrame(AFA_experiments)
for i in range(n_iterations):
    for j in range(10):
        GF_column = "Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1) 
        AFA_df[GF_column] = AFA_df[i][j][0]
AFA_df.drop([i for i in range(n_iterations)], axis = 1, inplace = True)

IID_df = pd.DataFrame(IID_experiments)
for i in range(n_iterations):
    for j in range(10):
        GF_column = "Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1) 
        IID_df[GF_column] = IID_df[i][j][0]
IID_df.drop([i for i in range(n_iterations)], axis = 1, inplace = True)



#Group Fairness experiments with IID summary:
IID_table_GF = [[i+1,j+1,IID_df["Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1)][j]] for i in range(n_iterations) for j in range(10)]
IID_GF_experiments = pd.DataFrame(IID_table_GF, columns=['Iteration','Experiment','Estimated_Group_Fairness'])
IID_GF_experiments["error"] = abs(IID_GF_experiments["Estimated_Group_Fairness"] - true_gf)
#Group Fairness experiments with AFA summary:
AFA_table_GF = [[i+1,j+1,AFA_df["Group_Fairness_iteration_" + str(i+1) + "_experiment_" + str(j+1)][j]] for i in range(n_iterations) for j in range(10)]
AFA_GF_experiments = pd.DataFrame(AFA_table_GF, columns=['Iteration','Experiment','Estimated_Group_Fairness'])
AFA_GF_experiments["error"] = abs(AFA_GF_experiments["Estimated_Group_Fairness"] - true_gf)

Sp_Summary['non-efficient_afa']['experiments'] = AFA_GF_experiments
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

AFA_experiments = [dict([(k, []) for k in range(n_iterations)]) for _ in range(10)]    
start_time = time()
for idx,experiment in enumerate(AFA_experiments):
    for i in range(n_iterations):
        experiment[i].append(gf_estimator((idx+1)*100))
end_time = time()
computation_time_eff_afa = end_time - start_time
Sp_Summary['efficient_afa']['computation_time_eff_afa'] = computation_time_eff_afa

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

with open('Compas_SP_RF_Experiments.pkl', 'wb') as handle:
    pickle.dump(Sp_Summary, handle)