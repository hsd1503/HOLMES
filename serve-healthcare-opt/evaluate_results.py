import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,precision_recall_curve,auc
import os

pred_dir = '/shared/choa/KDD_2020/pred_results/'
true_label_dir = '/shared/choa/KDD_2020/true_labels/'
record_test_file = '/shared/choa/KDD_2020/RECORD-test-test-shuffled'
record_test_1_file = '/shared/choa/KDD_2020/RECORD-test-test-1'
model_list = ['II_m{0}'.format(i) for i in range(1,17)]

def ReadLines(file, n=3000):
    list_files = []
    i = 0
    with open(file,'r') as f:
        for line in f:
            line = line.strip()
            list_files.append(line)
            i += 1
            if i > n:
                break
    return list_files

def evaluate_ensemble_models_per_patient(b, debug=False):
    record_test_list = ReadLines(record_test_1_file)

    roc_aucs = []
    pr_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    
    if debug:
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                    print(model_list[i]) 
    y_i_true = []
    y_i_scores = []
    last_pid = ''
    for record in record_test_list:
        record = record.replace('.pickle','')
        toks = record.split('_')
        cur_pid = toks[0]
        
        if not os.path.exists(true_label_dir + record + '.npy'):
            continue
            
        # get label
        y = np.load(true_label_dir + record + '.npy')
        
        # get ensemble scores
        y_scores = []
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                yhat = np.load(pred_dir + model_list[i] + '/' + record + '.npy')
                y_scores.append(yhat)
        
        y_scores = np.nan_to_num(np.array(y_scores))
                
        if len(y_scores) > 1:
            y_scores = np.mean(y_scores,0)
        else:
            y_scores = y_scores[0]
        
        # group by pid
        if cur_pid != last_pid:
            if debug:
                print(cur_pid)
                
            if len(y_i_true) > 0:
                y_i_true = np.array(y_i_true)
                y_i_scores = np.array(y_i_scores)
                if debug:
                    print(np.sum(y_i_true), len(y_i_true))
                if (np.sum(y_i_true) != 0) and (np.sum(y_i_true) < len(y_i_true)):
                    precision, recall, _ = precision_recall_curve(y_i_true, y_i_scores)
                    pr_auc = auc(recall, precision)
                    y_i_pred = (y_i_scores > 0.5)+ 0

                    roc_aucs.append(roc_auc_score(y_i_true, y_i_scores))
                    pr_aucs.append(pr_auc)
                    f1_scores.append(f1_score(y_i_true, y_i_pred))
                    precisions.append(precision_score(y_i_true,y_i_pred))
                    recalls.append(recall_score(y_i_true,y_i_pred))
            
            y_i_true = []
            y_i_scores = []
            last_pid = cur_pid
            
        y_i_true.extend(list(y))
        y_i_scores.extend(list(y_scores))
        
    if len(y_i_true) > 0:
        y_i_true = np.array(y_i_true)
        y_i_scores = np.array(y_i_scores)
        if (np.sum(y_i_true) != 0) and (np.sum(y_i_true) < len(y_i_true)):
            precision, recall, _ = precision_recall_curve(y_i_true, y_i_scores)
            pr_auc = auc(recall, precision)
            y_i_pred = (y_i_scores > 0.5)+ 0

            roc_aucs.append(roc_auc_score(y_i_true, y_i_scores))
            pr_aucs.append(pr_auc)
            f1_scores.append(f1_score(y_i_true, y_i_pred))
            precisions.append(precision_score(y_i_true,y_i_pred))
            recalls.append(recall_score(y_i_true,y_i_pred))
  
    return np.mean(roc_aucs), np.std(roc_aucs), np.mean(pr_aucs), np.std(pr_aucs), np.mean(f1_scores), np.std(f1_scores), np.mean(precisions), np.std(precisions), np.mean(recalls), np.std(recalls)

def evaluate_ensemble_models(b, num_record_eval = 100, debug=False):
    record_test_list = ReadLines(record_test_file, num_record_eval)
    
    y_true = []
    y_scores = []
    
    for record in record_test_list:
        record = record.replace('.pickle','')
        y = np.load(true_label_dir + record + '.npy')
        y_true.extend(list(y))
        
    for i in range(len(model_list)):
        if i == len(b):
            break
        if b[i] == 1:
            if debug:
                print(model_list[i])
            y_i_scores = []
            for record in record_test_list:
                record = record.replace('.pickle','')
                yhat = np.load(pred_dir + model_list[i] + '/' + record + '.npy')
                y_i_scores.extend(list(yhat))
            y_scores.append(y_i_scores)
    
    y_scores = np.nan_to_num(np.array(y_scores))
    
    if len(y_scores) == 0:
        return 0.0
    
    if len(y_scores) > 1:
        y_scores = np.mean(y_scores,0)
    else:
        y_scores = y_scores[0]
    if debug:
        print(len(y_true), len(y_scores))
        
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    y_pred = (y_scores > 0.5)+ 0
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), recall_score(y_true,y_pred),precision_score(y_true,y_pred),pr_auc
    


if __name__ == "__main__":
    # roc_auc,f1,recall,precision,pr_auc = evaluate_ensemble_models(b = [1,0], num_record_eval = 100, debug=True)
    # print(roc_auc,roc_auc,f1,recall,precision,pr_auc)
    roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std = evaluate_ensemble_models_per_patient(b = [1,0], debug=False)
    print(roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std)
