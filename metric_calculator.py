import json
import os
# import together_api
# from neurips2020utils.metric.metric_utils import compute_metric
# from together_api import dataset
from tqdm import tqdm
from evaluate import compute_score

def load_dataset_from_file(file_path):
    entries = []
    with open(file_path, "r") as f:
        if 'jsonl' not in f.name and 'json' in f.name:
            tmp_entries = json.load(f)
            for k,v in tmp_entries.items():
                v['index'] = k
                entries.append(v)
        else:
            for line in f:
                entries.append(json.loads(line))
    return entries

def calculate_full_metric(predictions, description=''):
    scores = []
    # calculate the final score for each prediction
    predictions_pbar = tqdm(predictions)
    predictions_pbar.set_description('processing ' + description)
    for prediction in predictions_pbar:
        k_full_score = compute_score([prediction['ground_truth']],
                                     prediction['predictions'],
                                     prediction['confidences'],
                                     {'u1':1.0, 'u2':1.0})
        scores.append(k_full_score)
    return sum(scores) / len(scores)


def total_accuracy(path, recalculate=False):
    invocation_prediction_gt = load_dataset_from_file(path)

    total_score = calculate_full_metric(invocation_prediction_gt, path)
    return total_score

if __name__ == '__main__':
    with open('results.txt', 'w') as benchmark:
        for root, subdirs, files in os.walk('./benchmarks'):
            # pbar = (files)
            for f in files:
                if 'tmp.' in f or '.placeholder' == f:
                    continue
                # pbar.set_description(f'processing file {os.path.join(root, f)}')
                file_path = os.path.join(root, f)
                acc = total_accuracy(file_path)
                print(f, '\t', acc)
                benchmark.write(f'{file_path}\t{acc}\n')

    # with open('./benchmarks/[gpt-4o-mini]-nl2bash-data-refactored.json', 'r') as benchmark:
    #     data = json.load(benchmark)
    #     together_api.dataset = together_api.get_dataset(os.getenv('DATASET_PATH'))
    #     total_score = []
    #     for k in tqdm(data.keys()):
    #         try:
    #             score = compute_metric(data[k]['prediction'], data[k]['confidence'], together_api.dataset[k]['cmd'])
    #             data[k]['score'] = score
    #             total_score.append(score)
    #         except Exception as e:
    #             print(e)
    # with open('./benchmarks/[gpt-4o-mini]-nl2bash-data-refactored.json', 'w') as benchmark:
    #     json.dump(data, benchmark)
    #     print(sum(total_score) / len(total_score))