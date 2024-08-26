import json
import os


def total_accuracy(path):
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
            for f in files:
                if 'tmp.' in f:
                    continue
                file_path = os.path.join(root, f)
                acc = total_accuracy(file_path)
                print(f, '\t', acc)
                benchmark.write(f'{file_path}\t{acc}\n')