"""
todo: use https://github.com/unslothai/unsloth to do inference and finetune of LLMs
maybe use https://github.com/ollama/ollama-python just for inference
maybe use https://github.com/abetlen/llama-cpp-python to use a very fast version of llama
"""
import math
from json import dumps

import tqdm

from neurips2020utils.metric.metric_utils import compute_metric
from utils.send_request import *
from utils.data_loader import get_dataset
# Example response from your server (shortened for clarity)

data_path_tellina = './data/nl2bash-data.json'

url = "http://localhost:8000/v1/completions"
headers = {
    "Content-Type": "application/json"
}

# Define the payload data
data = {
    "n_predict": 128,
    "temperature": 0,
    "logprobs": True,
    "max_tokens": 50,
    "seed": 438441351351443,
    "stop": ['<|im_end|>', '</s>', '\n',],
}

dataset_tellina = get_dataset(data_path_tellina)

def inference(prompt):
    data['prompt'] = prompt

    data_request = RequestData(**data)
    time_processed, response = chat_llama_cpp_server(url=url, headers=headers, data=data_request)

    # response.remove_backticks()
    # response.clean_text()

    # Extract tokens and log probabilities
    logprobs = response.logprobs
    tokens = response.tokens

    # Calculate confidence (probability) for each token
    confidences = [math.exp(logprob) for logprob in logprobs]

    average_confidence = -1 if len(confidences) == 0 else \
        sum(confidences) / len(confidences) # use the worst case if there is no prediction

    return response.text, average_confidence

def tellina_inference(model_name, notes=''):
    total_accuracy = []
    predictions = dict()
    with open(f'./benchmarks/{model_name}-{notes}-.tmp.json', 'a') as benchtmp:
        benchtmp.write('{')
        completition, total_entries = 0, len(dataset_tellina)
        for k, command in tqdm.tqdm(dataset_tellina.items()):
            k = int(k)

            nl = f'{command["invocation"]}. Bash command: '

            gtcmd = command['cmd']

            # print(f'processing {k}, completition {(completition/total_entries)*100:.4f}%')

            prediction, confidence = inference(nl)
            score = compute_metric(predicted_cmd=prediction, predicted_confidence=confidence, ground_truth_cmd=gtcmd)
            total_accuracy.append(score)

            predictions[k] = {
                'prediction': prediction,
                'confidence': confidence,
                'score': score,
                'ground_truth': gtcmd,
            }
            completition += 1
            benchtmp.write(f'"{k}": {dumps(predictions[k])},\n')
        benchtmp.write('}')
        with open(f'./benchmarks/{model_name}-{notes}-.json', 'w') as benchmark_tellina:
            json.dump(predictions, benchmark_tellina, indent=2)

    return sum(total_accuracy) / len(total_accuracy)


if __name__ == '__main__':
    total = tellina_inference('Meta-Llama-3-8B.Q8_0.gguf', notes='WITH_prompt_engineering')
    print('total accuracy', total)

# gtnl = 'Set permission of "file" to read only for the owner'
# gtcmd = 'find $HOME -name ".*" -ls'
#
# predcmd = 'find path -name ".*" -ls'
#
# score = compute_metric(predicted_cmd=predcmd, predicted_confidence=0.84135, ground_truth_cmd=gtcmd)
# print(score)
