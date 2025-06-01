import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import traceback
import math

import warnings
warnings.filterwarnings('ignore')

THRESHOLD_INDICATING_SHUTDOWN = 30
MISSING_VALUE = -999
OPERATION_VAL_RANGE = (713.682, 763.826)

def remove_long_shutdown(numbers, num_consecutive, missing_label):
    chunks = []
    current_chunk = []

    i = 0
    while i < len(numbers)-1:
        num = numbers[i]
        if num != missing_label:
            current_chunk.append(i)
        else:
            j = i+1
            while j < len(numbers):
                if numbers[j] == missing_label:
                    j += 1
                else:
                    break

            if j-i < num_consecutive:
                current_chunk += range(i,min(j+1, len(numbers)))# numbers[i:j+1]
            else:
                chunks.append(current_chunk)
                current_chunk = []

            i=j         

        i+= 1

    if current_chunk:
        chunks.append(current_chunk)

    to_ret = []
    for i, chunk in enumerate(chunks, 1):
        to_ret += chunk
        
    return to_ret

def align_data(result_df, label_df):
    first_valid_idx = None
    first_valid_value = None
    
    for i, observed_val in enumerate(result_df['observed']):
        if observed_val != MISSING_VALUE and not pd.isna(observed_val):
            first_valid_idx = i
            first_valid_value = observed_val
            break
    
    if first_valid_value is None:
        raise ValueError("No valid observed values found in result file")
    
    tolerance = 1e-3  
    
    matching_indices = []
    for i, label_value in enumerate(label_df['value']):
        if abs(label_value - first_valid_value) <= tolerance:
            matching_indices.append(i)
    
    if not matching_indices:
        differences = abs(label_df['value'] - first_valid_value)
        closest_idx = differences.idxmin()
        closest_diff = differences[closest_idx]
        
        if closest_diff < 1.0:
            matching_indices = [closest_idx]
        else:
            raise ValueError(f"Could not find suitable matching value for {first_valid_value} in label data")
    
    label_start_idx = matching_indices[0]
    
    result_aligned = result_df.iloc[first_valid_idx:].reset_index(drop=True)
    
    max_label_rows = len(label_df) - label_start_idx
    max_rows = min(len(result_aligned), max_label_rows)
    
    result_aligned = result_aligned.iloc[:max_rows]
    label_aligned = label_df.iloc[label_start_idx:label_start_idx + max_rows].reset_index(drop=True)
    
    
    for i in range(min(3, len(result_aligned))):
        result_val = result_aligned.iloc[i]['observed']
        label_val = label_aligned.iloc[i]['value']
        diff = abs(result_val - label_val)
    
    return result_aligned, label_aligned

import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import traceback

def calculate_f1(path_to_result, label_file, feasibility=None):
    for result_file in os.listdir(path_to_result):
        try:
            print('----------', result_file, '----------')
            result_adapad = pd.read_csv(path_to_result + '/' + result_file)
            result_adapad = result_adapad.dropna()
            result_adapad = result_adapad.reset_index(drop=True)

            label = pd.read_csv(label_file)
            
            result_adapad, label = align_data(result_adapad, label)
            
            if feasibility:
                label_feasibility = pd.read_csv(feasibility)
                cutoff_label_feasibility = label.timestamp.iloc[0] if 'timestamp' in label.columns else None
                if cutoff_label_feasibility is not None:
                    cutoff_idx = label_feasibility[label_feasibility.timestamp == cutoff_label_feasibility].index.values
                    if len(cutoff_idx) > 0:
                        label_feasibility = label_feasibility[cutoff_idx[0]:]
            
            total = pd.concat([result_adapad, label], axis=1, join='inner')
            if feasibility:
                total = total[:len(label_feasibility)]
                
            observed_values = total.observed.values.tolist()
            observed_values = [int(x) for x in observed_values]
            to_keep_comparision = remove_long_shutdown(observed_values, THRESHOLD_INDICATING_SHUTDOWN, MISSING_VALUE)
            total=total[total.index.isin(to_keep_comparision)]

            preds = result_adapad.anomalous
            preds = preds.dropna()
            preds = preds.astype(int)

            Precision, Recall, F, Support =metrics.precision_recall_fscore_support(total.is_anomaly.values.tolist(), 
                                                                                   total.anomalous.values.tolist(), 
                                                                                   zero_division=0)
            precision = Precision[1]
            recall = Recall[1]
            f = F[1]
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f:.4f}')
        except:
            traceback.print_exc()
        

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_roc(path_to_result, label_file, feasibility=None):
    for result_file in os.listdir(path_to_result):
        try:
            print('----------', result_file, '----------')
            result_adapad = pd.read_csv(path_to_result + '/' + result_file)
            result_adapad = result_adapad.dropna()
            result_adapad = result_adapad.reset_index(drop=True)

            label = pd.read_csv(label_file)
            
            result_adapad, label = align_data(result_adapad, label)
            
            if feasibility:
                label_feasibility = pd.read_csv(feasibility)
                cutoff_label_feasibility = label.timestamp.iloc[0] if 'timestamp' in label.columns else None
                if cutoff_label_feasibility is not None:
                    cutoff_idx = label_feasibility[label_feasibility.timestamp == cutoff_label_feasibility].index.values
                    if len(cutoff_idx) > 0:
                        label_feasibility = label_feasibility[cutoff_idx[0]:]

            total = pd.concat([result_adapad, label], axis=1, join='inner')
            if feasibility:
                total = total[:len(label_feasibility)]
            observed_values = total.observed.values.tolist()
            observed_values = [int(x) for x in observed_values]
            to_keep_comparision = remove_long_shutdown(observed_values, THRESHOLD_INDICATING_SHUTDOWN, MISSING_VALUE)
            total=total[total.index.isin(to_keep_comparision)]

            total["anomaly_score"] = np.ones((len(total), 1))
            normal_observed_range = total[(total.observed >= OPERATION_VAL_RANGE[0]) & 
                                          (total.observed <= OPERATION_VAL_RANGE[1])]
            anomaly_scores_normal_condition = normal_observed_range.err - normal_observed_range.threshold
            anomaly_scores_normal_condition = anomaly_scores_normal_condition.values.tolist()
            anomaly_scores_normal_condition = [sigmoid(x) for x in anomaly_scores_normal_condition]
            total.loc[normal_observed_range.index, 'anomaly_score'] = anomaly_scores_normal_condition

            roc_auc = metrics.roc_auc_score(total.is_anomaly, total.anomaly_score)
            print(f'ROC-AUC Score: {roc_auc:.4f}')

            y, x, _ = metrics.precision_recall_curve(total.is_anomaly, total.anomaly_score)
            pr_auc = metrics.auc(x, y)
            print(f'PR-AUC Score: {pr_auc:.4f}')
        except:
            traceback.print_exc()

path_to_result = "../results/Tide_pressure/"
label_file = "../data/Tide_Pressure.validation_stage.csv"
calculate_f1(path_to_result, label_file)
calculate_roc(path_to_result, label_file)