import math
import os
import time
from json import dumps, dump
from metric_calculator import calculate_full_metric

import utils.argparser as argparser
from tqdm import tqdm

from neurips2020utils.metric.metric_utils import compute_metric
from utils.data_loader import get_dataset
from utils.send_request import RequestData, inference_with_together_api, wait_rate_limit, inference_with_api

# rate_limit = 1
# use_proxy = False
# proxies = None
# dataset_path = ""
dataset = {}
# url = ""
prompt_format = ''

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": ''
}
data = {
    # "n_predict": 1,
    "temperature": 0.8,
    "logprobs": True,
    "max_tokens": 50,
    # "seed": 438441351351443,
    # "stop": ['<|im_end|>', '</s>', '\n'],
    "n":5,
    # "top_k":50, # TODO new
}

def inference(prompt):
    data['prompt'] = prompt

    # data_request = RequestData(**data)
    n_responses = inference_with_api(url=url, headers=headers, data=data, proxies=proxies)
    pass
    # response.remove_backticks()
    # response.clean_text()

    # Extract tokens and log probabilities
    n_logprobs = [response.logprobs for response in n_responses]
    n_texts = [response.text for response in n_responses]
    n_confidences = []
    # Calculate confidence (probability) for each token
    for logprobs in n_logprobs:
        confidences = [math.exp(logprob) for logprob in logprobs]

        average_confidence = -1 if len(confidences) == 0 else \
            sum(confidences) / len(confidences)  # use the worst case if there is no prediction

        n_confidences.append(average_confidence)

    return n_texts, n_confidences


def benchmark(model_name, notes='', base_prompt='', early_stop=None):
    tmpfile = f'./benchmarks/tmp.[{os.path.basename(model_name)}]-{os.path.basename(dataset_path)}-{notes}.json'
    outfile = f'./benchmarks/[{os.path.basename(model_name)}]-{os.path.basename(dataset_path)}-{notes}.json'
    total_accuracy = []
    predictions = dict()

    def resume_from_crash():
        """
        I'm sorry for whoever must read the following code,
        it works and does it well (with a high, but still not considerable, overhead).
        Do not question, even if I could, I am not able to answer for it.
        """
        import re
        import json
        with open(tmpfile, 'r') as resume:
            resume = resume.readlines()
            if len(resume) == 0:
                os.remove(tmpfile)
                return 0

            last_line = resume[-1]
            p = re.compile("\"[0-9]+\"")
            for i in reversed(range(len(resume))):
                l = resume[i].strip()
                if p.match(l):
                    last_line = resume[i]
                    break
            p = re.compile("[0-9]+")
            els = p.findall(last_line)
            if len(els) > 0:
                last = int(els[0])
            else:
                last = 0

            for l in resume:
                if len(l) == 0:
                    continue
                nl2cmd = f"{{{l.rpartition(',')[0]}}}" # https://stackoverflow.com/questions/15012228/splitting-on-last-delimiter-in-python-string
                key = p.findall(nl2cmd)
                if len(key) == 0:
                    continue
                key = key[0]
                nl2cmd = json.loads(nl2cmd)
                if len(nl2cmd) > 0:
                    # total_accuracy.append(nl2cmd[key]['score'])
                    predictions[key] = nl2cmd[key]
            return last

    data['model'] = model_name
    resume_from = 0
    try:
        resume_from = resume_from_crash() # 0 if there were no crashes
    except FileNotFoundError:
        print(f'Warning: no file {tmpfile} found') # non mi va di usare un logger ora
        with open(tmpfile, 'w'):
            pass
    # print(data['temperature'])
    stop_cond = 0
    with open(tmpfile, 'a+') as benchtmp:
        for k, command in tqdm(dataset.items()):
            k = int(k)
            if k <= resume_from:
                # skip the processed commands
                continue
            if early_stop is not None and stop_cond >= early_stop:
                break
            nl = base_prompt + prompt_format.format(command['invocation'])
            gtcmd = command['cmd']


            n_prediction, n_confidence = inference(nl)
            reset_limit = time.time()
            predictions[k] = {
                'predictions': [ ], # predictions
                'confidences': [ ], # confidences
                'scores':      [ ],
                'ground_truth':gtcmd,
            } # list of { gound_truth, prediction_list, confidence_list, scores_list }
            for prediction, confidence in zip(n_prediction, n_confidence):
                score = compute_metric(predicted_cmd=prediction, predicted_confidence=confidence, ground_truth_cmd=gtcmd)
                predictions[k]['scores'].append(score)
                predictions[k]['confidences'].append(confidence)
                predictions[k]['predictions'].append(prediction)

            benchtmp.write(f'"{k}": {dumps(predictions[k])},\n')
            time_passed = time.time() - reset_limit
            # wait_rate_limit(rate_limit - time_passed)

            if rate_limit >= 0:
                wait_rate_limit(rate_limit - time_passed)


            stop_cond += 1

        with open(outfile, 'w') as benchmark_tellina:
            dump(predictions, benchmark_tellina, indent=1)

    scores = calculate_full_metric(predictions)
    return scores


def setup_envs():
    # -------------------------------------------
    # scary area, pythonic jump scare!!!
    args = argparser.parser.parse_args()
    for arg in vars(args):
        var = str(arg).upper()
        value = str(getattr(args, str(arg)))
        if var != '' and  value != '':
            os.environ[var] = value # override env variables

    # beware, war crimes ahead
    # for arg in vars(args):
    #     os.environ[str(arg).upper()] = (
    #         os.environ)[str(arg).upper()] \
    #         if os.environ[str(arg).upper()] == '' \
    #         else str(getattr(args, str(arg))
    #     ) # override env variables
    # -------------------------------------------


setup_envs()
rate_limit = float(os.getenv('RATE_LIMIT'))  # one request for second :(
use_proxy = os.getenv('PROXY_PIA') if os.getenv('PROXY_PIA') != '' else False
dataset_path = os.getenv('DATASET_PATH')
url = os.getenv('URL')
model = os.getenv('MODEL_PATH')
notes = os.getenv('NOTES')
proxies = None if not use_proxy else {'all': os.getenv('PROXY_PIA')}
temp = os.getenv('TEMPERATURE')
try:
    data['temperature'] = float(temp)
except ValueError as e:
    print(e)
    data['temperature'] = 0.8
    r = input("Temperature is not a number, proceed with 0.8? [y]n")
    if r.lower() != 'y':
        exit(2)
    del r
    data['temperature'] = 0.8
del temp

headers['Authorization'] = f"Bearer {os.getenv('API_KEY')}"




if __name__ == '__main__':

    # with open('benchmarks/tmp.[Llama-3-8b-hf]-nl2bash-data.json.json', 'w' ) as f:
    #     with open("benchmarks/['meta-llama', 'Llama-3-8b-hf']-few-shots.json", 'r') as ff:
    #         r = json.load(ff)
    #         for k, v in r.items():
    #             f.write(f"{k}: {v},\n")
    #
    #
    base_prompt = '''Task: Convert the following descriptions into bash commands.\n'''
    prompt_format = 'Description: {} \nBash command: '
    dataset = get_dataset(dataset_path)
    if notes == 'part1':
        dataset = {k: v for k, v in dataset.items() if int(k) <= len(dataset) // 2}
    if notes == 'part2':
        dataset = {k: v for k, v in dataset.items() if int(k) > len(dataset) // 2}
    required = {'rate': rate_limit, 'proxy': use_proxy, 'dataset_path': dataset_path, 'url': url, 'model': model}
    if '' in required:
        raise Exception(f"some variable missing: {required}")
    del required

    total = benchmark(model, notes=notes, base_prompt=base_prompt)
    print('DONE:\n', total)
    # total = benchmark(model, notes=notes) # uncomment to use it without custom prompt


    # uncomment to show the accuracy of the first benchmark
    # outfile = f"./benchmarks/['meta-llama', 'Llama-3-8b-hf']-few-shots.json"
    # avg = total_accuracy(outfile)

    # print('total accuracy', avg)

    # ping proxy-nl.privateinternetaccess.com