import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from pathlib import Path
import os
import json
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import copy

from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from choa_evaluate_results_simplify import evaluate_ensemble_models, evaluate_ensemble_models_with_history_per_patient


def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_model(base_filters, n_block):
    model = ResNet1D(in_channels=1,
                    base_filters=base_filters,
                    kernel_size=16,
                    stride=2,
                    n_block=n_block,
                    groups=base_filters,
                    n_classes=2,
                    downsample_gap=max(n_block//8, 1),
                    increasefilter_gap=max(n_block//4, 1),
                    verbose=False)
    # print(model.get_info())
    return model

def get_description(n_gpu, n_patients, is_small=False):
    """
    return V and c

    V: base_filters, n_block, accuracy, flops, size
    c: n_gpu, n_patients
    """

    df = pd.read_csv('model_list.csv')
    if is_small:
        df = df.iloc[:12,:]
    print('model description:\n', df)
    # idx,model_idx,modality,n_filters,n_blocks,macs,max_memory_size,params(M),accuracy,latency
    # (0) n_filters, (1) n_blocks: build model
    # (2) model_idx: model architecture index
    # (3) max_memory_size: estimate model size
    # (4) accuracy, (5) latency
    V = df.loc[:, ['n_filters', 'n_blocks', 'model_idx', 'max_memory_size', 'accuracy', 'latency']].values
    c = np.array([n_gpu, n_patients])
    print(V)

    return V, c

def b2cnt(b, V):
    """
    b: binary list. fixed length for a model_list.csv.  [0,1,1,0,1,1]
    m=V['model_idx']: model list. the same length of b. [1,2,3,1,2,3]
    cnt: count list. length 20, used for store cache. above case: [0,2,2]+[0]*17. 
    """
    cnt = [0 for _ in range(20)]
    m = np.array(V[:,2], dtype=int)
    tmp_m = np.array(m)[np.array(b, dtype=bool)]
    cnter = Counter(tmp_m)
    for k, v in cnter.items():
        cnt[k] = v
    # print('b: ', b, 'cnt: ', cnt)
    return cnt

def cnt2b(cnt, V):
    """
    b: binary list. fixed length for a model_list.csv.  [0,1,1,0,1,1]
    m=V['model_idx']: model list. the same length of b. [1,2,3,1,2,3]
    cnt: count list. length 20, used for store cache. above case: [0,2,2]+[0]*17. 

    NOTE: this func is only used for precompute cache, since we don't know which model is counted
    """
    m = list(V[:,2])
    b = [0 for _ in range(len(m))]
    cnter = dict(zip(list(range(20)), cnt))
    for k, v in cnter.items():
        if v == 0:
            continue
        elif v == 1:
            b[m.index(k)] = 1
        elif v == 2:
            indices = [i for i, x in enumerate(m) if x == k]
            b[indices[0]] = 1
            b[indices[1]] = 1
        elif v == 3:
            indices = [i for i, x in enumerate(m) if x == k]
            b[indices[0]] = 1
            b[indices[1]] = 1
            b[indices[2]] = 1
    return b

def read_cache_accuracy():
    cache_accuracy = []
    with open('cache_accuracy.txt', 'r') as fin:
        for line in fin:
            content = line.strip().split('|')
            b = np.array([int(float(i)) for i in content[1].replace('[', '').replace(']', '').split(',')])
            accuracy = np.array([float(i.strip()) for i in content[2].replace('[', '').replace(']', '').split(',')])
            cache_accuracy.append([b, accuracy])
    return cache_accuracy

def read_cache_latency():
    cache_latency = []
    with open('cache_latency.txt', 'r') as fin:
        for line in fin:
            content = line.strip().split('|')
            cnt = np.array([int(float(i)) for i in content[1].replace('[', '').replace(']', '').split(',')])
            latency = float(content[2])
            cache_latency.append([cnt, latency])
    return cache_latency

def my_eval(gt, pred):
    out = []
    out.append(mean_absolute_error(gt, pred))
    out.append(r2_score(gt, pred))
    return out

def dist(v1, v2):
    return np.sum(np.abs(np.array(v1) - np.array(v2)))

# ------------------------------------------------------------------------------------------------
def get_accuracy_profile(V, b, cache, return_all=False):
    """
    choa
    """
    # print('profiling accuracy: ', b)
    local_b = copy.deepcopy(b)
    if len(local_b) != 60:
        local_b = local_b + [0]*(60-len(local_b))
    # print('profiling accuracy: ', len(local_b), local_b, b)

    if np.sum(local_b) == 0:
        final_accuracy = [0,0]
    else:
        if cache is not None:
            for i in cache:
                if dist(local_b, i[0]) == 0:
                    print('using accuracy cache for {}!'.format(local_b))
                    if return_all:
                        return i[1]
                    else:
                        return i[1][0]
            print('no accuracy cache found for {}!'.format(local_b))

        roc_auc, pr_auc, _, _ = evaluate_ensemble_models(local_b)
        final_accuracy = [roc_auc, pr_auc]

        if cache is not None:
            cache.append([list(local_b), final_accuracy])
        with open('cache_accuracy.txt', 'a') as fout:
            fout.write('{}|{}|{}\n'.format(get_now(), list(local_b), final_accuracy))

    if return_all:
        return final_accuracy
    else:
        return final_accuracy[0]

def get_latency_profile(V, c, b, cache, debug=False):
    """
    """
    # print('profiling latency: ', b)

    cnt = b2cnt(b, V)

    if debug:
        final_latency = 1e-3*np.random.rand()
    if np.sum(b) == 0:
        final_latency = 0.0
    else:
        # (1) use cache
        if cache is not None:
            for i in cache:
                if dist(cnt, i[0]) == 0:
                    print('using latencyy cache for {}!'.format(b))
                    return i[1]
            print('no latency cache found for {}!'.format(b))

        v = V[np.array(b, dtype=bool)]
        model_list = []
        total_size = 0
        for i_model in v:
            base_filters, n_block = int(i_model[0]), int(i_model[1])
            total_size += i_model[3]
            model_list.append(get_model(base_filters, n_block))
        model_size = total_size/1024
        print('The model size is: {:.4f} GB'.format(model_size))

        if model_size > 30:
            print('The model is too big')
            final_latency = 1e6 # set too large 1e6
        else:
            filename = "profile_results.jsonl"
            p = Path(filename)
            p.touch()
            os.environ["SERVE_PROFILE_PATH"] = str(p.resolve())
            file_path = Path(filename)
            system_constraint = {"gpu":int(c[0]), "npatient":int(c[1])}
            print(system_constraint)
            try:
                final_latency = profiler.profile_ensemble(model_list, file_path, system_constraint, fire_clients=False, with_data_collector=False)
            except Exception as e:
                print('Got error: ', e)
                final_latency = 1e6 # set exception 1e6
            if final_latency is None:
                print('Got None')
                final_latency = 1e6 # set None 1e6

        if cache is not None:
            cache.append([list(cnt), final_latency])

        if final_latency < 1e2:
            with open('cache_latency.txt', 'a') as fout:
                fout.write('{}|{}|{}\n'.format(get_now(), list(cnt), final_latency))

    return final_latency

# ------------------------------------------------------------------------------------------------
def test_fit_latency():
    """
    experiment to see latency proxy
    """
    cache_latency = read_cache_latency()
    B = []
    latency = []
    for i in cache_latency:
        B.append(i[0])
        latency.append(i[1])

    B = np.array(B)
    latency = np.array(latency)

    X_train, X_test, y_train, y_test = train_test_split(B, latency, test_size=0.9, random_state=0)
    print(X_train.shape, X_test.shape)

    m = RF()
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(mean_absolute_error(y_test, pred))
    print(r2_score(y_test, pred))


    plt.figure(figsize=(4,3))
    plt.plot([0,2],[0,2], 'r--')
    plt.scatter(y_test, pred, c='tab:grey', s=2)
    plt.xlabel('True Latency')
    plt.ylabel('Predicted Latency')
    plt.tight_layout()
    plt.savefig('img/cor_latency.png')

if __name__ == "__main__":

    # test b2cnt, cnt2b
    # V, c = get_description(1,1)
    # b = [0,1,1] + [0]*17 + [0,1,1] + [0]*17 + [0] * 20
    # cnt = [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # print(b2cnt(b, V))
    # print(cnt2b(cnt, V))

    # V, c = get_description(1,1, is_small=True)
    # b = [0,1,1,0] + [0,1,1,0] + [0,1,1,0]
    # cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0]
    # print(b2cnt(b, V))
    # print(cnt2b(cnt, V))
    
    V, c = get_description(1,1, is_small=False)

    # test get_accuracy_profile
    # b = [0,1,1,0] + [0,1,1,0] + [0,1,1,0]
    # b=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # b=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # b=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    b1 = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    b2 = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    b3 = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    b4 = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    b5 = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    
    print(b2cnt(b1, V))
    print(b2cnt(b2, V))
    print(b2cnt(b3, V))
    print(b2cnt(b4, V))
    print(b2cnt(b5, V))


    # # print(get_accuracy_profile(V, b, cache=None))
    # print(get_latency_profile(V, c, b, cache=None))

