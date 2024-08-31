import json
import math
import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

import together_api
from evaluate import compute_score, predict
import utils.argparser as argparser
from neurips2020utils.metric.metric_utils import compute_metric
from utils.data_loader import get_dataset
from utils.send_request import ResponseData, ResponseChatData

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

def reformat_gpt_batch_jsonl(entries, dataset, batch=''):
    import tqdm
    with open(f'./benchmarks/[gpt-4o-mini]-full.json', 'a+') as benchmark_tellina:
        for entry in tqdm.tqdm(entries):
            refactored_entries = dict()
            #  {"response": {"status_code": 200, "request_id": "64dcbb794474ca98c0111f35ec11368d", "body": {"id": "chatcmpl-A29yFOOKlC9IqR8kH7GTPath6znzh", "object": "chat.completion", "created": 1725078347, "model": "gpt-4o-mini-2024-07-18", "choices": [{"index": 0, "message": {"role": "assistant", "content": "`find . -type d -empty -delete`", "refusal": null}, "logprobs": {"content": [{"token": "`", "logprob": -2.5830445, "bytes": [96], "top_logprobs": []}, {"token": "find", "logprob": -3.1737043e-06, "bytes": [102, 105, 110, 100], "top_logprobs": []}, {"token": " .", "logprob": 0.0, "bytes": [32, 46], "top_logprobs": []}, {"token": " -", "logprob": -4.00813e-06, "bytes": [32, 45], "top_logprobs": []}, {"token": "type", "logprob": -5.080963e-06, "bytes": [116, 121, 112, 101], "top_logprobs": []}, {"token": " d", "logprob": -3.1281633e-07, "bytes": [32, 100], "top_logprobs": []}, {"token": " -", "logprob": -3.1281633e-07, "bytes": [32, 45], "top_logprobs": []}, {"token": "empty", "logprob": -0.0004614128, "bytes": [101, 109, 112, 116, 121], "top_logprobs": []}, {"token": " -", "logprob": 0.0, "bytes": [32, 45], "top_logprobs": []}, {"token": "delete", "logprob": -0.00044282433, "bytes": [100, 101, 108, 101, 116, 101], "top_logprobs": []}, {"token": "`", "logprob": 0.0, "bytes": [96], "top_logprobs": []}], "refusal": null}, "finish_reason": "stop"}, {"index": 1, "message": {"role": "assistant", "content": "```bash\nfind . -type d -empty -delete\n```", "refusal": null}, "logprobs": {"content": [{"token": "```", "logprob": -0.33304462, "bytes": [96, 96, 96], "top_logprobs": []}, {"token": "bash", "logprob": 0.0, "bytes": [98, 97, 115, 104], "top_logprobs": []}, {"token": "\n", "logprob": -1.504853e-06, "bytes": [10], "top_logprobs": []}, {"token": "find", "logprob": -6.869018e-06, "bytes": [102, 105, 110, 100], "top_logprobs": []}, {"token": " .", "logprob": -1.9361265e-07, "bytes": [32, 46], "top_logprobs": []}, {"token": " -", "logprob": -4.1273333e-06, "bytes": [32, 45], "top_logprobs": []}, {"token": "type", "logprob": -6.392203e-06, "bytes": [116, 121, 112, 101], "top_logprobs": []}, {"token": " d", "logprob": -1.9361265e-07, "bytes": [32, 100], "top_logprobs": []}, {"token": " -", "logprob": -6.704273e-07, "bytes": [32, 45], "top_logprobs": []}, {"token": "empty", "logprob": -0.0004730911, "bytes": [101, 109, 112, 116, 121], "top_logprobs": []}, {"token": " -", "logprob": 0.0, "bytes": [32, 45], "top_logprobs": []}, {"token": "delete", "logprob": -0.0005612541, "bytes": [100, 101, 108, 101, 116, 101], "top_logprobs": []}, {"token": "\n", "logprob": -5.5122365e-07, "bytes": [10], "top_logprobs": []}, {"token": "```", "logprob": -1.9361265e-07, "bytes": [96, 96, 96], "top_logprobs": []}], "refusal": null}, "finish_reason": "stop"}, {"index": 2, "message": {"role": "assistant", "content": "find . -type d -empty -delete", "refusal": null}, "logprobs": {"content": [{"token": "find", "logprob": -1.5830446, "bytes": [102, 105, 110, 100], "top_logprobs": []}, {"token": " .", "logprob": 0.0, "bytes": [32, 46], "top_logprobs": []}, {"token": " -", "logprob": -1.6524515e-05, "bytes": [32, 45], "top_logprobs": []}, {"token": "type", "logprob": -4.246537e-06, "bytes": [116, 121, 112, 101], "top_logprobs": []}, {"token": " d", "logprob": -3.1281633e-07, "bytes": [32, 100], "top_logprobs": []}, {"token": " -", "logprob": -1.147242e-06, "bytes": [32, 45], "top_logprobs": []}, {"token": "empty", "logprob": -0.0004102964, "bytes": [101, 109, 112, 116, 121], "top_logprobs": []}, {"token": " -", "logprob": 0.0, "bytes": [32, 45], "top_logprobs": []}, {"token": "delete", "logprob": -0.0005507679, "bytes": [100, 101, 108, 101, 116, 101], "top_logprobs": []}], "refusal": null}, "finish_reason": "stop"}, {"index": 3, "message": {"role": "assistant", "content": "`find . -type d -empty -delete`", "refusal": null}, "logprobs": {"content": [{"token": "`", "logprob": -2.5830445, "bytes": [96], "top_logprobs": []}, {"token": "find", "logprob": -3.1737043e-06, "bytes": [102, 105, 110, 100], "top_logprobs": []}, {"token": " .", "logprob": 0.0, "bytes": [32, 46], "top_logprobs": []}, {"token": " -", "logprob": -4.00813e-06, "bytes": [32, 45], "top_logprobs": []}, {"token": "type", "logprob": -5.080963e-06, "bytes": [116, 121, 112, 101], "top_logprobs": []}, {"token": " d", "logprob": -3.1281633e-07, "bytes": [32, 100], "top_logprobs": []}, {"token": " -", "logprob": -3.1281633e-07, "bytes": [32, 45], "top_logprobs": []}, {"token": "empty", "logprob": -0.0004614128, "bytes": [101, 109, 112, 116, 121], "top_logprobs": []}, {"token": " -", "logprob": 0.0, "bytes": [32, 45], "top_logprobs": []}, {"token": "delete", "logprob": -0.00044282433, "bytes": [100, 101, 108, 101, 116, 101], "top_logprobs": []}, {"token": "`", "logprob": 0.0, "bytes": [96], "top_logprobs": []}], "refusal": null}, "finish_reason": "stop"}, {"index": 4, "message": {"role": "assistant", "content": "```bash\nfind . -type d -empty -delete\n```", "refusal": null}, "logprobs": {"content": [{"token": "```", "logprob": -0.33304462, "bytes": [96, 96, 96], "top_logprobs": []}, {"token": "bash", "logprob": 0.0, "bytes": [98, 97, 115, 104], "top_logprobs": []}, {"token": "\n", "logprob": -1.504853e-06, "bytes": [10], "top_logprobs": []}, {"token": "find", "logprob": -6.869018e-06, "bytes": [102, 105, 110, 100], "top_logprobs": []}, {"token": " .", "logprob": -1.9361265e-07, "bytes": [32, 46], "top_logprobs": []}, {"token": " -", "logprob": -4.1273333e-06, "bytes": [32, 45], "top_logprobs": []}, {"token": "type", "logprob": -6.392203e-06, "bytes": [116, 121, 112, 101], "top_logprobs": []}, {"token": " d", "logprob": -1.9361265e-07, "bytes": [32, 100], "top_logprobs": []}, {"token": " -", "logprob": -6.704273e-07, "bytes": [32, 45], "top_logprobs": []}, {"token": "empty", "logprob": -0.0004730911, "bytes": [101, 109, 112, 116, 121], "top_logprobs": []}, {"token": " -", "logprob": 0.0, "bytes": [32, 45], "top_logprobs": []}, {"token": "delete", "logprob": -0.0005612541, "bytes": [100, 101, 108, 101, 116, 101], "top_logprobs": []}, {"token": "\n", "logprob": -5.5122365e-07, "bytes": [10], "top_logprobs": []}, {"token": "```", "logprob": -1.9361265e-07, "bytes": [96, 96, 96], "top_logprobs": []}], "refusal": null}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 81, "completion_tokens": 59, "total_tokens": 140}, "system_fingerprint": "fp_f33667828e"}}, "error": null}
            tmp_entry = entry['response']['body']
            id_ = entry['custom_id']
            refactored_entries[id_] = tmp_entry
            rds = []
            for i in range(len(tmp_entry['choices'])):
                rd = ResponseChatData(tmp_entry, i)
                rds.append(rd)

            n_responses = rds
            n_logprobs = [response.logprobs for response in n_responses]
            n_prediction = [response.text for response in n_responses]
            n_confidences = []
            # Calculate confidence (probability) for each token
            for logprobs in n_logprobs:
                confidences = [math.exp(logprob) for logprob in logprobs]

                average_confidence = -1 if len(confidences) == 0 else \
                    sum(confidences) / len(confidences)  # use the worst case if there is no prediction

                n_confidences.append(average_confidence)
            gtcmd = dataset[id_]['cmd']
            id_ = int(id_)
            predictions = {
                'index': id_,
                'predictions': [],  # predictions
                'confidences': [],  # confidences
                'scores': [],
                'ground_truth': gtcmd,
            }
            for prediction, confidence in zip(n_prediction, n_confidences):
                score = compute_metric(predicted_cmd=prediction, predicted_confidence=confidence, ground_truth_cmd=gtcmd)
                predictions[id_]['scores'].append(score)
                predictions[id_]['confidences'].append(confidence)
                predictions[id_]['predictions'].append(prediction)

                pred = json.dumps(predictions)
            benchmark_tellina.write(f'{pred}\n')
    # return n_prediction, n_confidences

def reformat_gpt_out():
    ds = get_dataset('./data/nl2bash-data.json')
    for i in range(1, 100):
        file = f'./benchmarks/[gpt-4o-mini-batch{i}]-nl2bash-zero-shot.jsonl'
        if not os.path.isfile(file):
            break
        gpt = load_dataset_from_file(file)
        reformat_gpt_batch_jsonl(gpt, ds, batch=str(i))


if __name__ == '__main__':
    # main()
    reformat_gpt_out()

