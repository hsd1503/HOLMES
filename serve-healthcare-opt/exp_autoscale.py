"""

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
from tqdm import tqdm
from datetime import datetime
import copy

from util import my_eval, get_accuracy_profile, get_latency_profile, read_cache_latency, read_cache_accuracy, get_description, get_now
from choa_evaluate_results_simplify import evaluate_ensemble_models, evaluate_ensemble_models_with_history_per_patient

np.random.seed(0)

##################################################################################################
# tools
##################################################################################################

def explore_genetic(n_model, B, n_samples, p, p1, dist=3):
    """
    Input:
        n_model: number of models, n
        n_samples: 

    Output:
        B \in \{0,1\}^{n_samples \times n_model}
    """
    B_local = copy.deepcopy(B)
    B_out = []
    n_samples_random = int(n_samples*(1-p))
    n_samples_mutation = int(n_samples*p*p1)
    n_samples_recombination = n_samples - n_samples_random - n_samples_mutation
    # print('n_samples_random:',n_samples_random,'n_samples_mutation:',n_samples_mutation,'n_samples_recombination:',n_samples_recombination)

    # print('B', len(B_local))
    B_random = explore_random(n_model, B_local, n_samples_random)
    # print(np.array(B_random))
    B_local = B_local + B_random
    B_out = B_out + B_random
    # print('B_random', len(B_local))
    B_mutation = explore_mutation(n_model, B_local, B_local, n_samples_mutation)
    # print(np.array(B_mutation))
    B_local = B_local + B_mutation
    B_out = B_out + B_mutation
    # print('B_mutation', len(B_local))
    B_recombination = explore_recombination(n_model, B_local, n_samples_recombination)
    # print(np.array(B_recombination))
    B_local = B_local + B_recombination
    B_out = B_out + B_recombination
    # print('B_recombination', len(B_local))
    # print(np.array(B_local))
    return B_out

def explore_random(n_model, B, n_samples):
    """
    Input:
        n_model: number of models, n
        n_samples: 

    Output:
        B \in \{0,1\}^{n_samples \times n_model}
    """
    global max_n_model
    print('use max_n_model={}'.format(max_n_model))
    out = []
    i = 0
    while i < n_samples:
        flag = True
        # get a random probability of 1s and 0s
        pp = (max_n_model/n_model)*np.random.rand()
        # get random binary vector
        tmp = np.random.choice([0, 1], size=n_model, p=(1-pp,pp))
        # dedup
        for b in out:
            if np.array_equal(tmp, b):
                flag = False
                break
        for b in B:
            if np.array_equal(tmp, b):
                flag = False
                break
        if flag: # found a valid sample
            out.append(list(tmp))
            i += 1
    return out

def explore_mutation(n_model, B_top, B, n_samples, dist=6):
    """
    """
    out = []
    i = 0
    while i < n_samples:
        flag = True
        # get a random b from B_top
        tmp = B_top[np.random.choice(list(range(len(B_top))))]
        tmp = copy.deepcopy(tmp)
        # random binary vector near dist, by random filp 3 digits
        for j in range(dist):
            idx = np.random.choice(list(range(n_model)))
            if tmp[idx] == 0:
                tmp[idx] = 1
            else:
                tmp[idx] = 0
        # dedup
        for b in out:
            if np.array_equal(tmp, b):
                flag = False
                break
        for b in B:
            if np.array_equal(tmp, b):
                flag = False
                break
        if flag: # found a valid sample
            out.append(list(tmp))
            i += 1
    return out

def explore_recombination(n_model, B, n_samples):
    out = []
    i = 0
    while i < n_samples:
        flag = True
        # get a random int from 1 to n_model-2
        split_idx = np.random.randint(1, n_model-1)
        # random pick two of b from B
        pick_idx1 = np.random.randint(0, len(B))
        pick_idx2 = np.random.randint(0, len(B))
        b1 = copy.deepcopy(B[pick_idx1])
        b2 = copy.deepcopy(B[pick_idx2])
        tmp = list(b1)[:split_idx] + list(b2)[split_idx:]
        # dedup
        for b in out:
            if np.array_equal(tmp, b):
                flag = False
                break
        for b in B:
            if np.array_equal(tmp, b):
                flag = False
                break
        if flag: # found a valid sample
            out.append(list(tmp))
            i += 1
    return out

# ---------------------------------------------------------------------------------------------------
def get_obj(accuracy, latency, lamda, L, soft=True):
    if soft:
        return accuracy + lamda * (L - latency)
    else:
        if latency > L:
            return 0.0
        else:
            return accuracy

def write_res(V, c, b, method):
    """
    profile and write a line
    """
    roc_auc, pr_auc, roc_outputs, pr_outputs = evaluate_ensemble_models(b=b)
    _, roc_auc_std, _, pr_auc_std, accuracy, accuracy_std, f1,f1_std,precision,precision_std,recall,recall_std = evaluate_ensemble_models_with_history_per_patient(b=b, obs_w_30sec=1, roc_outputs=roc_outputs, pr_outputs=pr_outputs, opt_cutoff=True, debug=False)
    latency = get_latency_profile(V, c, b, cache=cache_latency)

    b_out = [int(i) for i in b]
    with open(log_name, 'a') as fout:
        fout.write('{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.8f},{}\n'.format(c[0], c[1], method, roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy,accuracy_std, latency, b_out))

def write_traj(V, c, b, method):
    """
    more detailed results than write_res
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    roc_auc, pr_auc = get_accuracy_profile(V, b, cache=cache_accuracy, return_all=True)
    latency = get_latency_profile(V, c, b, cache=cache_latency)

    b_out = [int(i) for i in b]
    with open(traj_name, 'a') as fout:
        fout.write('{},{},{},{},{:.4f},{:.4f},{:.8f},{}\n'.format(get_now(), c[0], c[1], method, roc_auc, pr_auc, latency, b_out))

def write_proxy(mae_accuracy, r2_accuracy, mae_latency, r2_latency):
    with open(proxy_name, 'a') as fout:
        fout.write('{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(mae_accuracy, r2_accuracy, mae_latency, r2_latency))

##################################################################################################
# naive
##################################################################################################
def solve_random(V, c, L, lamda):
    """
    random incremental
    """
    print("="*60)
    print("start solve_random")
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_all = list(range(n_model))
    latency = 0.0
    for i in range(n_model):
        tmp_idx = np.random.choice(idx_all)
        idx_all.remove(tmp_idx)
        b[tmp_idx] = 1
        latency = get_latency_profile(V, c, b, cache=cache_latency)
        # print('model id: ', tmp_idx, 'b: ', b, latency)
        if latency >= L:
            b[tmp_idx] = 0
            break
        write_traj(V, c, b, 'solve_random')

    opt_b = b
    write_res(V, c, opt_b, 'solve_random')

    return opt_b

def solve_greedy_accuracy(V, c, L, lamda):
    """
    greedy accuracy incremental
    """
    print("="*60)
    print("start solve_greedy_accuracy")
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_order = np.argsort(V[:, 4])[::-1] # the 5th col is accuracy
    latency = 0.0
    for i in range(n_model):
        tmp_idx = idx_order[i]
        b[tmp_idx] = 1
        latency = get_latency_profile(V, c, b, cache=cache_latency)
        # print('model id: ', tmp_idx, 'b: ', b, latency)
        if latency >= L:
            b[tmp_idx] = 0
            break
        write_traj(V, c, b, 'solve_greedy_accuracy')
    print('found best b is: {}'.format(b))

    opt_b = b
    write_res(V, c, opt_b, 'solve_greedy_accuracy')

    return opt_b

def solve_greedy_latency(V, c, L, lamda):
    """
    greedy latency incremental
    """
    global max_n_model
    print("="*60)
    print("start solve_greedy_latency")
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_order = np.argsort(V[:, 5]) # the 6th col is latency
    latency = 0.0
    for i in range(n_model):
        tmp_idx = idx_order[i]
        b[tmp_idx] = 1
        latency = get_latency_profile(V, c, b, cache=cache_latency)
        # print('model id: ', tmp_idx, 'b: ', b, latency)
        if latency >= L:
            b[tmp_idx] = 0
            break
        write_traj(V, c, b, 'solve_greedy_latency')
    print('found best b is: {}'.format(b))
    max_n_model = int(np.sum(b))

    opt_b = b
    write_res(V, c, opt_b, 'solve_greedy_latency')

    return opt_b

##################################################################################################
# BO
##################################################################################################
def solve_opt_passive(V, c, L, lamda):

    global opt_b_solve_random
    global opt_b_solve_greedy_accuracy
    global opt_b_solve_greedy_latency

    print("="*60)
    print("start solve_opt_passive")
    # --------------------- hyper parameters ---------------------
    
    if global_debug:
        N1 = 1
    else:
        N1 = 100 # profile trials

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    try:
        B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
        print('warm init success')
        # print(B)
    except:
        print('warm init failed')
        if global_debug:
            B = []
        else:
            opt_b_solve_random = solve_random(V, c, L, lamda)
            opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
            opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)
            B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
    Y_accuracy = []
    Y_latency = []

    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency}

    # --------------------- (1) warm start ---------------------
    B = B + explore_random(n_model=n_model, B=B, n_samples=N1)
    # profile
    for b in tqdm(B):
        tmp_accuracy = get_accuracy_profile(V, b, cache=cache_accuracy)
        Y_accuracy.append(tmp_accuracy)
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        Y_latency.append(tmp_latency)
        write_traj(V, c, b, 'solve_opt_passive')

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        tmp = get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False)
        all_obj.append(tmp)
    # print(all_obj)
    opt_idx = np.argmax(np.nan_to_num(all_obj))
    opt_b = B[opt_idx]
    
    write_res(V, c, opt_b, 'solve_opt_passive')

    return opt_b

##################################################################################################
# AutoScale
##################################################################################################
def solve_proxy(V, c, L, lamda):

    global opt_b_solve_random
    global opt_b_solve_greedy_accuracy
    global opt_b_solve_greedy_latency

    print("="*60)
    print("start solve_proxy")
    # --------------------- hyper parameters ---------------------
    
    if global_debug:
        N1 = 3 # warm start
        N2 = 10 # proxy
        N3 = 3 # profile
        epoches = 1
    else:
        N1 = 0 # warm start
        N2 = 1000 # proxy
        N3 = 10 # profile trials in each epoch
        epoches = 10 # number of epoches

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    try:
        B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
        print('warm init success')
        # print(B)
    except:
        print('warm init failed')
        if global_debug:
            B = []
        else:
            opt_b_solve_random = solve_random(V, c, L, lamda)
            opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
            opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)
            B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
    Y_accuracy = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency}
    accuracy_predictor = RF(random_state=0)
    latency_predictor = RF(random_state=0)
    # print('len(B): ', len(B))

    # --------------------- (1) warm start ---------------------
    print('warm start')
    B = B + explore_random(n_model=n_model, B=B, n_samples=N1)
    # print('len(B) in warm start: ', len(B))
    # profile
    all_obj = []
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b, cache=cache_accuracy))
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        Y_latency.append(tmp_latency)
        # print('latency: ', Y_latency[-1])
        all_obj.append(get_obj(Y_accuracy[-1], Y_latency[-1], lamda, L, soft=False))
    tmp_opt_idx = np.argmax(np.nan_to_num(all_obj))
    write_traj(V, c, B[tmp_opt_idx], 'solve_proxy')

    # --------------------- (2) choose B ---------------------
    print('choose B start')
    for i_epoches in tqdm(range(epoches)):

        # fit proxy
        ## remove too large latency
        B_proxy = []
        Y_accuracy_proxy = []
        Y_latency_proxy = []
        for ii in range(len(Y_latency)):
            if Y_latency[ii] < 1e5:
                B_proxy.append(B[ii])
                Y_accuracy_proxy.append(Y_accuracy[ii])
                Y_latency_proxy.append(Y_latency[ii])
        print('remove {} too large latency profile'.format(len(B) - len(B_proxy)))
        accuracy_predictor.fit(B_proxy, Y_accuracy_proxy)
        latency_predictor.fit(B_proxy, Y_latency_proxy)

        pred_accuracy = accuracy_predictor.predict(B_proxy)
        pred_latency = latency_predictor.predict(B_proxy)
        mae_accuracy, r2_accuracy = my_eval(Y_accuracy_proxy, pred_accuracy)
        mae_latency, r2_latency = my_eval(Y_latency_proxy, pred_latency)
        write_proxy(mae_accuracy, r2_accuracy, mae_latency, r2_latency)

        # genetic explore
        # random: 1-p, mutation: p*p1
        B_bar = explore_genetic(n_model=n_model, B=B, n_samples=N2, p=0.5, p1=0.5)
        # print('len(B_bar): ', len(B_bar))
        pred_accuracy = accuracy_predictor.predict(np.array(B_bar))
        pred_latency = latency_predictor.predict(np.array(B_bar))
        all_obj = []
        for i in range(len(B_bar)):
            all_obj.append(get_obj(pred_accuracy[i], pred_latency[i], lamda, L, soft=False))
        top_idx = np.argsort(all_obj)[::-1][:N3]
        B_0 = list(np.array(B_bar)[top_idx])
        # for i in range(len(B_0)):
        #     print(top_idx[i], B_0[i])
        # print('len(B_0): ', len(B_0))

        # profile
        for b in B_0:
            # get_accuracy_profile
            Y_accuracy.append(get_accuracy_profile(V, b, cache=cache_accuracy))
            # get_latency_profile
            tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
            Y_latency.append(tmp_latency)
            # print('latency: ', Y_latency[-1])
            write_traj(V, c, b, 'solve_proxy')

        B = B + B_0
        # print('len(B) after epoch: ', len(B))
        all_obj = []
        for i in range(len(B)):
            all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False))
        tmp_opt_idx = np.argmax(np.nan_to_num(all_obj))
        write_traj(V, c, B[tmp_opt_idx], 'solve_proxy')

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False))
    opt_idx = np.argmax(np.nan_to_num(all_obj))
    opt_b = B[opt_idx]
    write_traj(V, c, opt_b, 'solve_proxy')
    write_res(V, c, opt_b, 'solve_proxy')

    return opt_b

if __name__ == "__main__":
    
    # L = 10 # maximum latency

    for L in [0.15, 0.25, 0.3]:

        lamda = 1
        tag = '60models_latency{}'.format(L)

        global max_n_model
        global_debug = False
        is_small = False

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_latency = read_cache_latency()
        cache_accuracy = read_cache_accuracy()
        log_name = 'res/log_{}_{}.txt'.format(current_time, tag)
        traj_name = 'res/traj_{}_{}.txt'.format(current_time, tag)
        proxy_name = 'res/proxy_{}_{}.txt'.format(current_time, tag)

        with open(log_name, 'w') as fout:
            fout.write(current_time+'\n')
        
        V, c = get_description(n_gpu=1, n_patients=1, is_small=is_small)
        print('model description:\n', V, '\nsystem description:', c)

        # # ---------- naive solutions ----------
        opt_b_solve_random = solve_random(V, c, L, lamda)
        opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
        opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)

        # # ---------- BO solution ----------
        opt_b_solve_opt_passive = solve_opt_passive(V, c, L, lamda)

        # ---------- AutoScale solution ----------
        opt_b_solve_proxy = solve_proxy(V, c, L, lamda)


