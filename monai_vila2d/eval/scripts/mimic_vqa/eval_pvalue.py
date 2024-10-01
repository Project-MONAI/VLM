# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
from evaluate_metrics import calculate_f1score
from glossary import normalize_word
from scipy.stats import wilcoxon
from utils import load_json, load_jsonl


def get_significance(instruct, prediction1, prediction2):
    gt = load_json(instruct)
    pred1 = load_jsonl(prediction1)
    pred2 = load_jsonl(prediction2)

    gt_ids = [item['id'] for item in gt]
    pred_ids1 = [item['question_id'] for item in pred1]
    pred_ids2 = [item['question_id'] for item in pred2]
    assert gt_ids == pred_ids1, "please make sure pred and gt are exactly matched"
    assert gt_ids == pred_ids2, "please make sure pred and gt are exactly matched"

    metrics1 = {'id': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': [], 'total': 0}
    metrics2 = {'id': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': [], 'total': 0}

    for gt_item, pred_item1, pred_item2 in zip(gt, pred1, pred2):
        if gt_item['id'] != pred_item1['question_id']:
            print(f"gt id: {gt_item['id']} pred id: {pred_item1['question_id']}")
            raise ValueError("id mismatch")
        elif gt_item['id'] != pred_item2['question_id']:
            print(f"gt id: {gt_item['id']} pred id: {pred_item2['question_id']}")
            raise ValueError("id mismatch")
        else:
            gt_value = gt_item['conversations'][1]['value'].lower()
            pred_value1 = pred_item1['text'].split('Assistant: ')[-1].lower().strip()
            pred_value2 = pred_item2['text'].split('Assistant: ')[-1].lower().strip()
            gt_value = normalize_word(gt_value)
            pred_value1 = normalize_word(pred_value1)
            pred_value2 = normalize_word(pred_value2)

            metrics1['total'] += 1
            metrics2['total'] += 1

            if gt_item['answer_type'] == 'closed':
                metrics1['accuracy'].append(int(gt_value == pred_value1))
                metrics2['accuracy'].append(int(gt_value == pred_value2))
            else:    
                f1_score1, precision1, recall1 = calculate_f1score(pred_value1, gt_value)
                f1_score2, precision2, recall2 = calculate_f1score(pred_value2, gt_value)
                

                metrics1['f1'].append(f1_score1)
                metrics1['precision'].append(precision1)
                metrics1['recall'].append(recall1)
                metrics1['id'].append(pred_item1['question_id'])
                metrics2['f1'].append(f1_score2)
                metrics2['precision'].append(precision2)
                metrics2['recall'].append(recall2)
                metrics2['id'].append(pred_item2['question_id'])
    
    return metrics1, metrics2

def print_stats(question_type, size, mean1, mean2, std1=0.0, std2=0.0):
    print(
        f"Metrics: {question_type} {size} " 
        f"Open {mean1*100:.2f} ({std1*100:.2f}) " 
        f"Close {mean2*100:.2f} ({std2*100:.2f})"
    )

# usage 
# define file paths
gt_root = Path('../groundtruth_vqa_json')
ckpt_root = Path('../predicted_vqa_json')

model1 = 'llava_med_slake_all_mimic'
model2 = 'llava_med_slake_all_mimic_expert'
filename1 = 'llava_med_slake_all_mimic'
filename2 = 'llava_med_slake_all_mimic_expert'

avg_open1 = 0
avg_open2 = 0
avg_close1 = 0
avg_close2 = 0
n_open = 0
n_close = 0

avg_opens1 = np.zeros(0)
avg_opens2 = np.zeros(0)
avg_closes1 = np.zeros(0)
avg_closes2 = np.zeros(0)

question_types = ['abnormality', 'presence', 'view', 'location', 'level', 'type']
# loop through question types
for ind, qtype in enumerate(question_types):
    gt = gt_root / qtype / 'llava_med_instruct_mimicvqa_test_type.json'
    accs1 = np.zeros(0)
    recalls1 = np.zeros(0)
    accs2 = np.zeros(0)
    recalls2 = np.zeros(0)

    for run in [4, 5, 6]:
        prediction1 = ckpt_root / model1 / qtype / f'{filename1}_run{run}.jsonl'
        prediction2 = ckpt_root / model2 / qtype / f'{filename2}_run{run}.jsonl'
        metrics1, metrics2 = get_significance(gt, prediction1, prediction2)

        accs1 = np.append(accs1, np.array(metrics1['accuracy']))
        recalls1 = np.append(recalls1, np.array(metrics1['recall']))

        accs2 = np.append(accs2, np.array(metrics2['accuracy']))
        recalls2 = np.append(recalls2, np.array(metrics2['recall']))
        total1 = metrics1['total']
        total2 = metrics2['total']

        avg_opens1 = np.append(avg_opens1, np.array(metrics1['recall']))
        avg_opens2 = np.append(avg_opens2, np.array(metrics2['recall']))
        avg_closes1 = np.append(avg_closes1, np.array(metrics1['accuracy']))
        avg_closes2 = np.append(avg_closes2, np.array(metrics2['accuracy']))
    
    if recalls1.shape[0] > 0:
        w1, p1 = wilcoxon(recalls1, recalls2, alternative='two-sided', method='auto')
    else: 
        w1, p1 = 0, 1
    if accs1.shape[0] > 0:
        w2, p2 = wilcoxon(accs1, accs2, alternative='two-sided', method='auto')
    else:
        w2, p2 = 0, 1

    recalls1mean = recalls1.mean() if recalls1.shape[0] > 0 else 0
    recalls2mean = recalls2.mean() if recalls2.shape[0] > 0 else 0
    accs1mean = accs1.mean() if accs1.shape[0] > 0 else 0
    accs2mean = accs2.mean() if accs2.shape[0] > 0 else 0
    
    run_size = int(recalls1.shape[0] / 3)
    if run_size > 0:
        recalls1std = [recalls1[i*run_size:(i+1)*run_size].mean() for i in range(3)]
        recalls1std = np.array(recalls1std).std()
    else:
        recalls1std = 0

    run_size = int(recalls2.shape[0] / 3)
    if run_size > 0:
        recalls2std = [recalls2[i*run_size:(i+1)*run_size].mean() for i in range(3)]
        recalls2std = np.array(recalls2std).std()
    else:
        recalls2std = 0

    run_size = int(accs1.shape[0] / 3)
    if run_size > 0:
        accs1std = [accs1[i*run_size:(i+1)*run_size].mean() for i in range(3)]
        accs1std = np.array(accs1std).std()
    else:
        accs1std = 0
    
    run_size = int(accs2.shape[0] / 3)
    if run_size > 0:
        accs2std = [accs2[i*run_size:(i+1)*run_size].mean() for i in range(3)]
        accs2std = np.array(accs2std).std()
    else:
        accs2std = 0

    n_open += recalls1.shape[0]
    avg_open1 += recalls1mean * recalls1.shape[0]
    avg_open2 += recalls2mean * recalls2.shape[0]
    n_close += accs1.shape[0]
    avg_close1 += accs1mean * accs1.shape[0]
    avg_close2 += accs2mean * accs2.shape[0]

    print("Model No Expert")
    print_stats(qtype, total1, recalls1mean, accs1mean, recalls1std, accs1std)
    print("Model Expert")
    print_stats(qtype, total2, recalls2mean, accs2mean, recalls2std, accs2std)
    print(f"Wilcoxon Open {p1:.4f} Close {p2:.4f}")

if avg_opens1.shape[0] > 0:
    w1, p1 = wilcoxon(avg_opens1, avg_opens2, alternative='two-sided', method='auto')
else:
    w1, p1 = 0, 1
if avg_closes1.shape[0] > 0:
    w2, p2 = wilcoxon(avg_closes1, avg_closes2, alternative='two-sided', method='auto')
else:
    w2, p2 = 0, 1

print("Model No Expert")
print_stats('average', n_open, avg_open1 / n_open, avg_close1 / n_close)
print("Model Expert")
print_stats('average', n_open, avg_open2 / n_open, avg_close2 / n_close)
print(f"Wilcoxon Open {p1:.4f} Close {p2:.4f}")

