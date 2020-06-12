import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm

from util import my_eval, get_accuracy_profile, get_latency_profile, get_description, get_now, b2cnt, cnt2b, read_cache_latency
from choa_evaluate_results_simplify import evaluate_ensemble_models, evaluate_ensemble_models_with_history_per_patient

def cnt_cache_latency(cache):

    outname = 'cache_latency.txt'
    V, c = get_description(n_gpu=1, n_patients=1, is_small=True)

    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    cnt = [0]*16 + [i1, i2, i3, i4]
                    b = cnt2b(cnt, V)
                    tmp_latency = get_latency_profile(V, c, b, cache=cache)

def b_cache_accuracy(cache):

    outname = 'cache_accuracy.txt'
    V, c = get_description(n_gpu=1, n_patients=1, is_small=True)

    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    cnt = [0]*16 + [i1, i2, i3, i4]
                    b = cnt2b(cnt, V)
                    final_res = get_accuracy_profile(V, b, cache=cache, return_all=True)

def precompute():
    """
    this code is for testing accuracy profile
    """
    out_fname = 'precompute_accuracy.txt'
    V, c = get_description(n_gpu=1, n_patients=1, is_small=False)
    n_model = V.shape[0]
    for i in range(n_model):
        b = np.zeros(n_model)
        b[i] = 1
        roc_auc, pr_auc, roc_outputs, pr_outputs = evaluate_ensemble_models(b=b)
        ret = evaluate_ensemble_models_with_history_per_patient(b=b, obs_w_30sec=1, roc_outputs=roc_outputs, pr_outputs=pr_outputs, opt_cutoff=True, debug=False)
        tmp_accuracy = get_accuracy_profile(V, b, cache=None, return_all=True)
        # tmp_latency = get_latency_profile(V, c, b, cache=None)
        print(tmp_accuracy)
        print(ret)
        # with open(out_fname, 'a') as fout:
        #     fout.write('{}\n'.format(tmp_accuracy))

if __name__ == "__main__":

    # cache_latency = read_cache_latency()
    # cnt_cache_latency(cache=cache_latency)

    # cache_accuracy = read_cache_accuracy()
    # cnt_cache_accuracy(cache=cache_accuracy)

    # the biggest
    # V, c = get_description(n_gpu=1, n_patients=1, is_small=True)
    # b = [1] * 12
    # final_res = get_latency_profile(V, c, b, cache=None)

    precompute()

