import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,precision_recall_curve,roc_curve,auc,accuracy_score
import os

Clinic_Evaluate = True

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

def ReadLabels(label_file, start_max, end_max):
    true_labels = []
    start_hrs = []
    end_hrs = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            toks = line.split(':')
            true_labels.append(int(toks[0]))
            start_hrs.append(min(int(toks[1]),start_max))
            end_hrs.append(min(int(toks[2]),end_max))
            
    return true_labels, start_hrs, end_hrs       
    
pred_dir = '../waveforms/'
model_list = ReadLines(pred_dir + 'model_list.txt')

file_labels = [0,1,1,1,1,1,1,1,0,1]

if Clinic_Evaluate:
    true_labels, start_hrs, end_hrs = ReadLabels(pred_dir + 'labels.txt', start_max = 24*3, end_max = 24*2)
else:
    true_labels = [0,1,1,1,1,1,1,1,0,1]
    start_hrs = [48 for i in range(len(true_labels))]
    end_hrs = [24 for i in range(len(true_labels))]

def evaluate_ensemble_models(b, debug=False):
    num_per_hr = 60*2 # 30 seconds
    
    y_all_score = []
    y_all_true = []
    
    if debug:
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                    print(model_list[i]) 
                    
    for n in range(len(true_labels)):
        # get ensemble scores
        y_scores = []
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                yhat = np.loadtxt('{0}{1}/{1}_{2}.{3}.out'.format(pred_dir, n, model_list[i], int(file_labels[n])))
                y_scores.append(yhat)
        
        y_scores = np.nan_to_num(np.array(y_scores))
                
        if len(y_scores) > 1:
            y_scores = np.mean(y_scores,0)
        else:
            y_scores = y_scores[0]
        
        y_all_score.extend(list(y_scores[:(start_hrs[n] * num_per_hr)]))
        y_all_true.extend(list(np.zeros(start_hrs[n] * num_per_hr)))
        
        if true_labels[n] == 1:
            y_all_score.extend(list(y_scores[-(end_hrs[n] * num_per_hr):]))
            y_all_true.extend(list(np.ones(end_hrs[n] * num_per_hr)))
    
    fpr, tpr, roc_thresholds = roc_curve(y_all_true, y_all_score)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_all_true, y_all_score)
    pr_auc = auc(recall, precision)
  
    return roc_auc, pr_auc, [fpr, tpr, roc_thresholds], [precision, recall, pr_thresholds]

def evaluate_ensemble_models_with_history(b, obs_w_30sec=1, debug=False):
    num_per_hr = 60*2 # 30 seconds
    
    y_all_score = []
    y_all_true = []
    
    if debug:
        print('Observation window {0} x 30 seconds'.format(obs_w_30sec))
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                    print(model_list[i]) 
                    
    for n in range(len(true_labels)):
        # get ensemble scores
        y_scores = []
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                yhat = np.loadtxt('{0}{1}/{1}_{2}.{3}.out'.format(pred_dir, n, model_list[i], int(file_labels[n])))
                y_scores.append(yhat)
        
        y_scores = np.nan_to_num(np.array(y_scores))
                
        if len(y_scores) > 1:
            y_scores = np.mean(y_scores,0)
        else:
            y_scores = y_scores[0]
        
        if obs_w_30sec == 1:
            y_all_score.extend(list(y_scores[:(start_hrs[n] * num_per_hr)]))
            y_all_true.extend(list(np.zeros(start_hrs[n] * num_per_hr)))

            if true_labels[n] == 1:
                y_all_score.extend(list(y_scores[-(end_hrs[n] * num_per_hr):]))
                y_all_true.extend(list(np.ones(end_hrs[n] * num_per_hr)))
        else:
            y_agg_scores = []
            tmp_scores = y_scores[:(start_hrs[n] * num_per_hr)]
            for s in range(0,start_hrs[n] * num_per_hr-obs_w_30sec):
                y_agg_scores.append(np.mean(tmp_scores[s:(s+obs_w_30sec)])) 
               
            y_all_score.extend(y_agg_scores)
            y_all_true.extend(list(np.zeros(start_hrs[n] * num_per_hr-obs_w_30sec)))
            
            if true_labels[n] == 1:
                y_agg_scores = []
                tmp_scores = y_scores[-(end_hrs[n] * num_per_hr):]
                for s in range(0,end_hrs[n] * num_per_hr - obs_w_30sec):
                    y_agg_scores.append(np.mean(tmp_scores[s:(s+obs_w_30sec)]))
                
                y_all_score.extend(y_agg_scores)
                y_all_true.extend(list(np.ones(end_hrs[n] * num_per_hr-obs_w_30sec)))
            
    fpr, tpr, roc_thresholds = roc_curve(y_all_true, y_all_score)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_all_true, y_all_score)
    pr_auc = auc(recall, precision)
  
    return roc_auc, pr_auc, [fpr, tpr, roc_thresholds], [precision, recall, pr_thresholds]


def evaluate_ensemble_models_with_history_per_patient(b, obs_w_30sec =1, roc_outputs=[], pr_outputs=[], opt_cutoff=True, debug=False):
    num_per_hr = 60*2 # 30 seconds
    
    if opt_cutoff:
        # compute opt cutoffs
        fpr, tpr, roc_thresholds = roc_outputs[0],roc_outputs[1],roc_outputs[2]
        opt_idx = np.argmax(tpr - fpr)
        acc_opt_cutoff = roc_thresholds[opt_idx]

        precision, recall, pr_thresholds = pr_outputs[0],pr_outputs[1],pr_outputs[2]
        recall += 0.001
        opt_idx = np.argmax((precision * recall)/(precision + recall))
        f1_opt_cutoff = pr_thresholds[opt_idx]
    else:
        acc_opt_cutoff = 0.5
        f1_opt_cutoff = 0.5
    
    roc_aucs = []
    pr_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    if debug:
        print('Observation window {0} x 30 seconds'.format(obs_w_30sec))
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                    print(model_list[i]) 
                    
    for n in range(len(true_labels)):
        # only evaluate label 1
        if true_labels[n] == 0:
            continue

        # get ensemble scores
        y_scores = []
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                yhat = np.loadtxt('{0}{1}/{1}_{2}.{3}.out'.format(pred_dir, n, model_list[i], int(file_labels[n])))
                y_scores.append(yhat)
        
        y_scores = np.nan_to_num(np.array(y_scores))
                
        if len(y_scores) > 1:
            y_scores = np.mean(y_scores,0)
        else:
            y_scores = y_scores[0]
        
        if obs_w_30sec == 1:
            y_agg_scores = np.concatenate([y_scores[:(start_hrs[n] * num_per_hr)], y_scores[-(end_hrs[n] * num_per_hr):]])
            y_true = np.concatenate([np.zeros(start_hrs[n] * num_per_hr), np.ones(end_hrs[n] * num_per_hr)])
        else:
            y_agg_scores = []
            
            tmp_scores = y_scores[:(start_hrs[n] * num_per_hr)]
            for s in range(0,start_hrs[n] * num_per_hr-obs_w_30sec):
                y_agg_scores.append(np.mean(tmp_scores[s:(s+obs_w_30sec)])) 
                
            tmp_scores = y_scores[-(end_hrs[n] * num_per_hr):]
            for s in range(0,end_hrs[n] * num_per_hr - obs_w_30sec):
                y_agg_scores.append(np.mean(tmp_scores[s:(s+obs_w_30sec)]))
                
            y_true = np.concatenate([np.zeros(start_hrs[n] * num_per_hr-obs_w_30sec), np.ones(end_hrs[n] * num_per_hr-obs_w_30sec)])
        
        y_scores = np.array(y_agg_scores)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_aucs.append(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_aucs.append(auc(recall, precision))
        
        y_acc_pred = (y_scores > acc_opt_cutoff)+ 0
        accuracies.append(accuracy_score(y_true, y_acc_pred))

        y_f1_pred = (y_scores > f1_opt_cutoff)+ 0
        f1_scores.append(f1_score(y_true, y_f1_pred))
        precisions.append(precision_score(y_true,y_f1_pred))
        recalls.append(recall_score(y_true,y_f1_pred))
            
  
    return np.mean(roc_aucs), np.std(roc_aucs), np.mean(pr_aucs), np.std(pr_aucs),np.mean(accuracies), np.std(accuracies), np.mean(f1_scores), np.std(f1_scores), np.mean(precisions), np.std(precisions), np.mean(recalls), np.std(recalls)

if __name__ == "__main__":

    # AF
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # LF
    # b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # ours
    # b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    roc_auc, pr_auc, roc_outputs, pr_outputs = evaluate_ensemble_models(b=b,debug=False)
    print(evaluate_ensemble_models_with_history_per_patient(b=b, obs_w_30sec=1, roc_outputs=roc_outputs, pr_outputs=pr_outputs, opt_cutoff=True, debug=False))
    # print(roc_auc, pr_auc)

    # #roc_auc, pr_auc, roc_outputs, pr_outputs = evaluate_ensemble_models_with_history(b=[1,1], obs_w_30sec = 10,debug=True)
    # #print(roc_auc, pr_auc)

    # roc_auc, roc_auc_std, pr_auc, pr_auc_std, accuracy, accuracy_std, f1,f1_std,precision,precision_std,recall,recall_std= evaluate_ensemble_models_with_history_per_patient(b=b, obs_w_30sec=1, roc_outputs = roc_outputs, pr_outputs = pr_outputs, opt_cutoff = True, debug=True)
    # print('Cutoff optimized')
    # print(roc_auc, roc_auc_std, pr_auc, pr_auc_std, accuracy, accuracy_std, f1,f1_std,precision,precision_std,recall,recall_std)

    # roc_auc, roc_auc_std, pr_auc, pr_auc_std, accuracy, accuracy_std, f1,f1_std,precision,precision_std,recall,recall_std= evaluate_ensemble_models_with_history_per_patient(b=b, obs_w_30sec=1, roc_outputs = roc_outputs, pr_outputs = pr_outputs, opt_cutoff = False, debug=True)
    # print('Cutoff not optimized')
    # print(roc_auc, roc_auc_std, pr_auc, pr_auc_std, accuracy, accuracy_std, f1,f1_std,precision,precision_std,recall,recall_std)
