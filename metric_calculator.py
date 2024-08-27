import json
import os
import together_api
from neurips2020utils.metric.metric_utils import compute_metric
from together_api import dataset
from tqdm import tqdm

def total_accuracy(path, recalculate=False):
    with open(path, 'r') as benchmark_tellina:
        data = {}
        if 'jsonl' in path:
            lines = benchmark_tellina.readlines()
            for l in lines:
                chunk = json.loads(l)
                k = chunk['index']
                del chunk['index']
                data[k] = chunk
        elif 'json' in path:
            data = json.load(benchmark_tellina)

    total_score = 0
    for k,v in data.items():
        total_score += v['score']
    return total_score / len(data)

if __name__ == '__main__':
    with open('results.txt', 'w') as benchmark:
        for root, subdirs, files in os.walk('./benchmarks'):
            for f in tqdm(files):
                if 'tmp.' in f:
                    continue
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