import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from evaluate import compute_score
import utils.argparser as argparser
args = argparser.parser.parse_args()

def load_dataset_from_file(file_path):
    entries = []
    with open(file_path, "r") as f:
        if 'jsonl' not in f.name and 'json' in f.name:
            tmp_entries = json.load(f)
            for k, v in tmp_entries.items():
                v['index'] = k
                entries.append(v)
        else:
            for line in f:
                entries.append(json.loads(line))
    return entries

def calculate_full_metric(predictions, description=''):
    scores = []
    predictions_pbar = tqdm(predictions, desc='Processing ' + description)
    for prediction in predictions_pbar:
        k_full_score = compute_score(
            [prediction['ground_truth']],
            prediction['predictions'],
            prediction['confidences'],
            {'u1': 1.0, 'u2': 1.0}
        )
        scores.append(k_full_score)
    return sum(scores) / len(scores)

def total_accuracy(path, recalculate=False):
    invocation_prediction_gt = load_dataset_from_file(path)
    total_score = calculate_full_metric(invocation_prediction_gt, path)
    return total_score

def process_file(file_path):
    """Worker funxction to process a single file and return its accuracy."""
    acc = total_accuracy(file_path)
    return file_path, acc

def main():
    num_processes = args.processes if args.processes >= 0 else cpu_count() # Set number of concurrent processes

    # Collect all file paths to process
    file_paths = []
    for root, subdirs, files in os.walk('./benchmarks'):
        for f in files:
            if 'tmp.' in f or '.placeholder' == f:
                continue
            file_paths.append(os.path.join(root, f))

    # Process files in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, file_paths)

    # Write results to file after all processing is complete
    with open('results.txt', 'w') as benchmark:
        for file_path, acc in results:
            print(file_path, '\t', acc)
            benchmark.write(f'{file_path}\t{acc}\n')

if __name__ == '__main__':
    main()
