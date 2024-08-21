import json
import os
from openai import OpenAI

import together_api
from together_api import dataset_path
from utils.data_loader import get_dataset
import math
from time import sleep


# Define your CHAT_PROMPT as a string (placeholder)
CHAT_PROMPT = """You are a professional bash command writer.
Your sole task is to write a bash command based on the description provided by the user.
Your output must be strictly the command itself, no explanations or additional information are allowed—only the command.
"""

# Initialize the OpenAI client with the API key, project, and organization
client = OpenAI(api_key=os.environ['API_KEY'], project='proj_Jj0U3h7ovNTNr1eR33lWjlbv', organization='org-poN8ibmByghdh4thRUeuixFy')

n_count = 5


def batch_inference(dataset, start_from=0, stop_at=-1):
    """
    {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    """
    JSONL_PATH = './nl2cmd_batch.jsonl'
    entries = []
    if stop_at < 0:
        stop = len(dataset)

    def to_jsonline(entry_):
        return json.dumps(entry_)

    def dump_jsonl(jsonl):
        with open(JSONL_PATH, 'w') as f:
            for index, line in enumerate(jsonl):
                f.write(line + ('\n' if index < len(jsonl) - 1 else ''))

    current = 0
    truncate_dataset = list(dataset.items())[start_from:stop_at]
    dataset = dict(truncate_dataset)
    for key, invocation_cmd in dataset.items():
        prompt = (together_api.base_prompt +
                  together_api.prompt_format.format(invocation_cmd['invocation']))
        entry = {
            'custom_id': key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                'model':"gpt-4o",
                'messages':[
                    {"role": "system", "content": CHAT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                'logprobs':True,
                'max_tokens':128,
                'n':n_count,
                'temperature':0.8, # see https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
            }
        }
        entry = to_jsonline(entry)
        entries.append(entry)
        # print(entry)
    dump_jsonl(entries)
    batch_input_file = client.files.create(
        file=open(JSONL_PATH, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Tellina nl2bash benchmark job"
        }
    )
    with open('batch-obj.json', 'w') as f:
        json.dump(batch.to_dict(), f, indent=2)

    return batch.id


def retrieve_batch(file_id):
    file_response = client.files.content(file_id)
    print(file_response.text)


# Make a request to the OpenAI API
def inference(prompt):
    # data_request = RequestData(**data)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CHAT_PROMPT},
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        max_tokens=128,
        n=n_count,
        temperature=1.25, # see https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature

    )
    texts = response.choices
    predictions, confidences = [], []
    for text in texts:
        fix_text = text.message.content.replace('```bash\n', '').replace('```', '').strip()
        predictions.append(fix_text)
        # calculate confidence score for each text
        confidence = []
        for log_prob in text.logprobs.content:
            prob = math.exp(log_prob.logprob)
            confidence.append(prob)
        average_confidence = -1 if len(confidence) == 0 else \
            sum(confidence) / len(confidence)  # use the worst case if there is no prediction
        confidences.append(average_confidence)

    return predictions, confidences


together_api.inference = inference

# quit()


# print(inference('hello, nice to meet you.'))
if __name__ == '__main__':
    together_api.prompt_format = 'description: {}'
    together_api.base_prompt = "Write me a one line bash command for the following tasks, "

    together_api.dataset = get_dataset(dataset_path)
    # benchmark(model_name='gpt-4o-mini', base_prompt=together_api.base_prompt, early_stop=10)
    start= 4500
    window = 750
    stop = start + window
    batch_number = 7
    while start < len(together_api.dataset):
        batch_id = batch_inference(together_api.dataset, start_from=start, stop_at=stop)
        response = client.batches.retrieve(batch_id)
        while response.status != 'completed':
            if response.status == 'finalizing':
                print(f"{response.status}, waiting 10 seconds")
                sleep(10) # wait 10 seconds then try to retrieve the results
            else:
                print(f"{response.status}, waiting 3 minutes. Status: ", response.status)
                sleep(60*3) # wait 3 minutes then try to retrieve the results
            response = client.batches.retrieve(batch_id)
        file_response = client.files.content(response.output_file_id)
        output = file_response.text
        with open(f'./benchmarks/[gpt-4o-batch{batch_number}]-tellina.jsonl', 'w') as f:
            f.write(output)
        batch_number += 1
        start += window
        stop += window
    # Print the response from the model
    # print(completion.to_json())
