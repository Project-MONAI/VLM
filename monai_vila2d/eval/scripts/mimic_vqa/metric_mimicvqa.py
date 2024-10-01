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


import os
import json
# import jsonlines
import argparse

from pathlib import Path
import numpy as np

from glossary import normalize_word
from evaluate_metrics import calculate_f1score
from utils import load_json, load_jsonl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def get_metrics(instruct, prediction):
    gt = load_json(instruct)
    pred = load_jsonl(prediction)

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
    print(f'num_gt_ids: {num_gt_ids} num_pred_ids: {num_pred_ids}')
    # assert gt_ids == pred_ids, f"please make sure pred ({len(pred_ids)}) and gt ({len(gt_ids)}) are exactly matched"

    # gt_dict = {item['id'] : item for item in gt}
    pred_dict = {item['question_id'] : item for item in pred}
    
    metrics = {'id': [], 'f1': [], 'precision': [], 'recall': [], 'correct': 0, 'total_closed': 0, 'total_open': 0}

    for gt_item in gt:

        item_id = gt_item['id']
        pred_item = pred_dict[item_id]

        gt_value = gt_item['conversations'][1]['value'].lower()
        pred_value = pred_item['text'].split('Assistant: ')[-1].lower().strip()
        pred_value = normalize_word(pred_value)
        gt_value = normalize_word(gt_value)

        if gt_item['answer_type'] == 'closed':
            if gt_value == pred_value:
                metrics['correct'] += 1
                # print(f"gt: {gt_value} pred: {pred_value}")
            metrics['total_closed'] += 1
        else:    
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            metrics['f1'].append(f1_score)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['id'].append(pred_item['question_id'])
            metrics['total_open'] += 1

    return metrics


def main():

    args = get_args()

    gt_root = Path(args.input)
    ckpt_root = Path(args.answers)

    filename = 'llava_med_instruct_mimicvqa_test_type_answers_images_small'

    # loop through question types
    question_types = ['abnormality', 'presence', 'view', 'location', 'level', 'type']

    total_count_closed, total_count_open = np.zeros(0), np.zeros(0)
    open_recalls, closed_accs = np.zeros(0), np.zeros(0)
    results = dict()
    for qtype in question_types:
        gt = gt_root / qtype / 'llava_med_instruct_mimicvqa_test_type.json'
        accs = np.zeros(0)
        recalls = np.zeros(0)
        count_closed = np.zeros(0)
        count_open = np.zeros(0)
        for run in [0]:
            #prediction = ckpt_root / model / qtype / f'{filename}_run{run}.jsonl'    # Children's
            prediction = ckpt_root / f'{qtype}_{filename}.jsonl'    # TODO: possible use different runs
            metrics = get_metrics(gt, prediction)


            accuracy = metrics['correct'] / metrics['total_closed'] if metrics['total_closed'] > 0 else 0
            f1 = sum(metrics['f1']) / len(metrics['f1']) if len(metrics['f1']) > 0 else 0
            precision = sum(metrics['precision']) / len(metrics['precision']) if len(metrics['precision']) > 0 else 0
            recall = sum(metrics['recall']) / len(metrics['recall']) if len(metrics['recall']) > 0 else 0


            count_closed = np.append(count_closed, metrics['total_closed'])
            count_open = np.append(count_open, metrics['total_open'])
            accs = np.append(accs, accuracy)
            recalls = np.append(recalls, recall)

            assert len(np.unique(count_closed)) == 1, "count_closed was different between runs!"
            assert len(np.unique(count_open)) == 1, "count_open was different between runs!"

            #print(f"{qtype} run{run}: {metrics['correct']}/{metrics['total_closed']} open {recall:.4f} close {accuracy:.4f}")

        if count_closed.mean() > 0:
            closed_accs = np.append(closed_accs, accs.mean())
            total_count_closed = np.append(total_count_closed, count_closed.mean())
        if count_open.mean() > 0:
            open_recalls = np.append(open_recalls, recalls.mean())
            total_count_open = np.append(total_count_open, count_open.mean())

        results[qtype] = {"open": recalls.mean(), "closed": accs.mean()}
        print(f"{qtype} avg open: {100*results[qtype]['open']:.3f} close: {100*results[qtype]['closed']:.3f}")

    results["total_open"] = np.sum(total_count_open)
    results["total_closed"] = np.sum(total_count_closed)
    results["total_qas"] = np.sum(total_count_open) + np.sum(total_count_closed)
    print(f"Total open: {results['total_open']:.0f}", )
    print(f"Total closed: {results['total_closed']:.0f}")
    print(f"Total QAs: {results['total_qas']:.0f}")

    # Averages
    results["avg_open"] = np.average(open_recalls)
    results["avg_closed"] = np.average(closed_accs)
    print(f"Avg. open: {100*results['avg_open']:.3f}")
    print(f"Avg. closed: {100*results['avg_closed']:.3f}")
    results["wavg_open"] = np.average(open_recalls, weights=total_count_open)
    results["wavg_closed"] = np.average(closed_accs, weights=total_count_closed)    # TODO: use for eval
    print(f"wAvg. open: {100*results['wavg_open']:.3f}")
    print(f"wAvg. closed: {100*results['wavg_closed']:.3f}")

    with open(args.output, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
