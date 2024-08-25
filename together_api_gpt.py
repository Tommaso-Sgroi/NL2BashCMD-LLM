import math
import os
import time
from json import dumps, dump

from tqdm import tqdm

from neurips2020utils.metric.metric_utils import compute_metric
from utils.data_loader import get_dataset
from utils.send_request import RequestData, inference_with_together_api

url = "https://api.together.xyz/v1/completions"
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}"
}
data = {
    # "n_predict": 1,
    "temperature": 0,
    "logprobs": True,
    "max_tokens": 50,
    "seed": 438441351351443,
    "stop": ['<|im_end|>', '</s>', '\n', '\\'],
}
rate_limit = 1 # one request for second :(
use_proxy = True
dataset_path = "data/magnum_chatGPT_generated_data.json"
# dataset_path = "./data/nl2bash-data.json"
dataset = get_dataset(dataset_path)

def inference(prompt):
    data['prompt'] = prompt

    # data_request = RequestData(**data)
    response = inference_with_together_api(url=url, headers=headers, data=data, use_proxy=use_proxy)
    # response.remove_backticks()
    # response.clean_text()

    # Extract tokens and log probabilities
    logprobs = response.logprobs
    tokens = response.tokens

    # Calculate confidence (probability) for each token
    confidences = [math.exp(logprob) for logprob in logprobs]

    average_confidence = -1 if len(confidences) == 0 else \
        sum(confidences) / len(confidences)  # use the worst case if there is no prediction

    return response.text, average_confidence


def benchmark(model_name, notes='', base_prompt=''):
    tmpfile = f'./benchmarks/{model_name.replace("/", "")}-{notes}-.tmp.json'
    outfile = f'./benchmarks/{model_name.split("/")}-{notes}-.json'
    total_accuracy = []

    def wait_rate_limit(rate):
        time.sleep(rate + 0.2)

    def resume_from_crash():
        import re
        import json
        with open(tmpfile, 'r') as resume:
            resume = resume.readlines()[1:]
            last_line = resume[-1]
            p = re.compile("[0-9]+")
            els = p.findall(last_line)
            if len(els) > 0:
                last = int(els[0])
            else:
                last = 0

            for l in resume:
                nl2cmd = f"{{{l.rpartition(',')[0]}}}" # https://stackoverflow.com/questions/15012228/splitting-on-last-delimiter-in-python-string
                nl2cmd = json.loads(nl2cmd)
                total_accuracy.append(nl2cmd['score'])
            return last

    predictions = dict()
    data['model'] = model_name
    resume_from = resume_from_crash() # 0 if there were no crashes
    with open(tmpfile, 'a') as benchtmp:
        benchtmp.write('{\n')
        for k, command in tqdm(dataset.items()):
            k = int(k)
            if k < resume_from:
                # skip the processed commands
                continue
            nl = base_prompt + f"Description: {command['invocation']} \nBash command: "
            gtcmd = command['cmd']


            prediction, confidence = inference(nl)
            reset_limit = time.time()
            score = compute_metric(predicted_cmd=prediction, predicted_confidence=confidence, ground_truth_cmd=gtcmd)
            total_accuracy.append(score)

            predictions[k] = {
                'prediction': prediction,
                'confidence': confidence,
                'score': score,
                'ground_truth': gtcmd,
            }
            benchtmp.write(f'"{k}": {dumps(predictions[k])},\n')
            time_passed = time.time() - reset_limit
            # print(rate_limit - time_passed)
            wait_rate_limit(rate_limit - time_passed)
            # c += 1
            # if c == 5:
            #     break
        benchtmp.write('}')
        with open(outfile, 'w') as benchmark_tellina:
            dump(predictions, benchmark_tellina, indent=2)

    return sum(total_accuracy) / len(total_accuracy)




if __name__ == '__main__':
    base_prompt = '''Task: Convert the following descriptions into bash commands.
Description: Remove all files in the current directory.
Bash Command: rm -rf *

Description: List all files in a directory including hidden files.
Bash Command: ls -a

Description: Create a new directory called "backup".
Bash Command: mkdir backup

Description: Check the current disk usage.
Bash Command: df -h

Description: Find all ".txt" files in the current directory.
Bash Command: find . -name "*.txt"

Description: Find all ".sh" files in the current directory.
Bash Command: find . | grep "sh"

Description: Copy a file called "example.txt" to the "backup" directory.
Bash Command: cp example.txt backup/

Description: list all files that ends with the .txt extension in the current directory.
Bash Command: ls -a | grep .txt

'''
    total = benchmark('meta-llama/Llama-3-8b-hf', notes='WITH_HIGH_prompt_engineering-GPT_Dataset', base_prompt=base_prompt)
    print('total accuracy', total)

