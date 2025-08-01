
import math

import numpy as np

from scipy import stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score

import sys


def get_mse(y,f):
  
    y = np.array(y)
    f = np.array(f)
    
    mse = ((y - f)**2).mean(axis=0)

    return mse


def get_cindex(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    CI = round(CI, 4)
    return CI


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))



def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    
    if (upp == 0) or (down == 0):
        return 1

    return 1 - (upp / float(down))



def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    fenmu = float(y_obs_sq * y_pred_sq)
    if fenmu == 0:
        fenmu = 1
    return mult / fenmu


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    output = r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
    output = round(output, 4)
    return output


def positive(y_true):
    return np.sum((y_true == 1))

def negative(y_true):
    return np.sum((y_true == 0))

def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))

def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))

def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true) + 1e-9
    return tp / p

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true) + 1e-9
    return tn / n

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca)
    return fs

auc_score = roc_auc_score
kappa_score = cohen_kappa_score

def spearman(y,f):
    

    rs = stats.spearmanr(y, f)[0]

    return rs

from lifelines.utils import concordance_index

def get_cindex_mgraphdta(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    if np.sum(gt_mask) == 0:
        CI =  np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / (np.sum(gt_mask)+1e-10)
    else:
        CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI


def get_concordance_index(Y, P):
    return concordance_index(Y, P)



def get_metrics_classification(real_score, predict_score):
    real_score = np.array(real_score)
    predict_score = np.array(predict_score)
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [round(auc[0, 0], 4), round(aupr[0, 0], 4), round(f1_score, 4),
            round(accuracy, 4), round(recall, 4), round(specificity, 4),
            round(precision, 4)]

def get_metrics(task, y_true, y_pred, threshold=0.5):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if task == 'Classification':
        temp = get_metrics_classification(y_true, y_pred)
        auc, aupr, f1_score, accuracy, recall, specificity, precision= temp
        metrics = {
            'auc': auc, 
            'aupr': aupr, 
            'f1_score': f1_score, 
            'accuracy': accuracy, 
            'recall': recall,
            'specificity': specificity, 
            'precision': precision
        }
    elif task == 'Regression':
        
        mse = get_mse(y_true, y_pred)
        # ci = get_concordance_index(y_true, y_pred)
        try:
            ci = get_cindex_mgraphdta(y_true, y_pred)
        except:
            ci = 0.5
        rm2 = get_rm2(y_true, y_pred)
        metrics = {'mse': mse, 
                   'ci': ci,
                   'rm2': rm2}
    else:
        sys.exit(5)
    return metrics



