import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

import pandas as pd
import copy


def evaluate_dp(h, X_a1, X_a0):
    pos_rate_a1 = np.average(h.predict(X_a1))
    pos_rate_a0 = np.average(h.predict(X_a0))
    return pos_rate_a1- pos_rate_a0

def generate_random_model(X, h_star, seed=1):
    '''
    Random labels
    '''
    #np.random.seed(seed)
    rand_y = np.random.random(len(X)) > 0.5
    return LinearSVC(penalty='l2', loss="hinge", max_iter=10000).fit(X, rand_y)

def get_consistent_model(Sx, Sy, max_exp=25):
    '''
    Generate high enough weights to ensure fitted model is in version space
    Minimal regularization
    '''
    for i in range(5, max_exp):
        max_iter = 5000 + 5 ** i
        weights = 5 ** i * np.ones(len(Sx))
        clf = LinearSVC(penalty='l2', loss="hinge",
                        max_iter=max_iter, C=1e-6).fit(Sx, Sy, sample_weight=weights)
        if np.all(clf.predict(Sx) == Sy):
            return clf
    return

def label_diff(h_hat, h1, h2, X):
    '''
    Computes \delta(h_hat, h_1, h_2)
    '''
    d1 = np.where(h_hat.predict(X) != h1.predict(X))[0]
    d2 = np.where(h_hat.predict(X) != h2.predict(X))[0]
    d12 = list(set(d1).union(set(d2)))
    return d12

def train_hinge_model(full_X, full_y, X1, X0, Sx, beta, lmbd1=1., lmbd0=1.):
    '''
    Train weighted hinge loss under linear kernel
    '''
    weights = np.concatenate([np.ones(len(X1)), np.ones(len(X0)), beta * np.ones(len(Sx))])
    clf = LinearSVC(penalty='l2', loss="hinge", max_iter=50000).fit(full_X, full_y, sample_weight=weights)
    return clf

def binary_search_lmbd(full_X, full_y, X1, X0, Sx, Sy):
    
    '''
    Find large enough lmbd to allow for complete fit
    Generally max value is around 500-10000
    '''
    for i in range(5, 25):
        lmbd = 5 ** i
        clf = train_hinge_model(full_X, full_y, X1, X0, Sx, beta=lmbd)
        if np.all(clf.predict(Sx) == Sy):
            return lmbd, clf
    
    return None, None

def train_opt_mu(Sx, Sy, X_a1, X_a0, radius=10, increment=10):
    
    all_X = np.vstack([X_a1, X_a0, Sx])
    y11 = np.ones(X_a1.shape[0])
    y10 = np.zeros(X_a1.shape[0])
    y01 = np.ones(X_a0.shape[0])
    y00 = np.zeros(X_a0.shape[0])
    
    max_dp_y = np.concatenate([y11, y00, Sy])
    min_dp_y = np.concatenate([y10, y01, Sy])
    
    clfs = []
    print("search around near-opt lambda for max DP and min DP")

    '''
    Opt lambda for max DP and min DP
    '''
    lmbd_opt1, clf1 = binary_search_lmbd(all_X, max_dp_y, X_a1, X_a0, Sx, Sy)
    if clf1 is not None:
        lmbd_range1 = [np.abs(lmbd_opt1 + increment * i) for i in range(-radius, radius + 1)]
        clfs.extend([train_hinge_model(all_X, max_dp_y, X_a1, X_a0, Sx, beta=lmbd) for lmbd in lmbd_range1])
        clfs.append(clf1)
    
    lmbd_opt2, clf2 = binary_search_lmbd(all_X, min_dp_y, X_a1, X_a0, Sx, Sy)
    if clf2 is not None:
        lmbd_range2 = [np.abs(lmbd_opt2 + increment * i) for i in range(-radius, radius + 1)]
        clfs.extend([train_hinge_model(all_X, min_dp_y, X_a1, X_a0, Sx, beta=lmbd) for lmbd in lmbd_range2])
        clfs.append(clf2)
    
    '''
    Keep only models in the version space
    '''
    consistent_clfs = []
    for clf in clfs:
        if np.all(clf.predict(Sx) == Sy): consistent_clfs.append(clf)

    return consistent_clfs

def optimize_mu(Sx, Sy, X_a1, X_a0, radius=10):
    hs = train_opt_mu(Sx, Sy, X_a1, X_a0, radius=radius)
    if len(hs) == 0: return None, None, None, None
    dps = np.array([evaluate_dp(h, X_a1, X_a0) for h in hs])
    max_i = np.argmax(dps)
    min_i = np.argmin(dps)
    return hs[max_i], hs[min_i], dps[max_i], dps[min_i]

def init_weights(X, delta=0.1):
    '''
    Initialize weights/threshold
    '''
    m = X.shape[0]
    ws = np.ones(m) / m
    vc_dim = X.shape[1] #linear model
    rate = 2 * vc_dim * np.log(m) + np.log(1 / delta)
    txs = np.random.exponential(scale=1./rate, size=(m,))
    return ws, txs

def list_diff(main_list, sub_list):
    return np.array(list(set(main_list).difference(set(sub_list))))

#################################################################################

def exceed_budget(exceeded, S_T, label_budget):
    print("exceeded budget")
    _, S = cal_helper(S_T, exceeded, label_budget)
    h_hat = consistent(X[S], y[S])
    return evaluate_dp(h_hat, X_a1, X_a0), S

def under_budget(S_T, label_budget, implicit_labeled):
    remain_num = label_budget - len(S_T)
    remain_idx = list_diff(range(len(X)), S_T + implicit_labeled)
    
    _, S = cal_helper(S_T + implicit_labeled, remain_idx, label_budget)
    
    h_hat = consistent(X[S], y[S])
    return evaluate_dp(h_hat, X_a1, X_a0), S

def iid_sample(X, num):
    '''
    Sampling with replacement
    '''
    idxs = np.random.choice(len(X), num, replace=True)
    return X[idxs]

def iid_baseline(label_budget):
    group_size = int(label_budget / 2)
    X_a1_sample = iid_sample(X_a1, group_size)
    X_a0_sample = iid_sample(X_a0, group_size)
    return evaluate_dp(h_star, X_a1_sample, X_a0_sample)

def passive_learn_baseline(label_budget):
    idxs = []
    while True:
        idxs = np.random.choice(len(X), label_budget, replace=False)
        if len(set(y[idxs])) == 2: break
    h = consistent(X[idxs], y[idxs])
    return evaluate_dp(h, X_a1, X_a0)

#################################################################################
#################################################################################

def train_logistic_regression(X, y, rand_state=1, offset=True):
    clf = LogisticRegression(penalty='l2', random_state=rand_state, 
                             solver="saga", fit_intercept=offset).fit(X, y)
    w = clf.coef_.reshape(-1)
    w = np.append(w, clf.intercept_)
    return w

def append_offset(X):
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    return X1

def label_linear_dataset(w, X):
    return (X @ w >= 0).astype("int")

def evaluate_linear(w, X, y):
    return np.average(label_linear_dataset(w, X)  == y)

