import json
import os
from openai import OpenAI
from webview import start

import together_api
from together_api import benchmark, dataset_path
from utils.data_loader import get_dataset
import math
# Set up fake API keys, project name, and organization for the example
# os.environ['API_KEY'] = "fake-api-key-1234567890abcdef"
# os.environ['PROJECT_NAME'] = ""
# os.environ['ORGANIZATION_ID'] = "org-fakeorg123456"

# Define your CHAT_PROMPT as a string (placeholder)
CHAT_PROMPT = """You are a professional bash command writer.
Your sole task is to write a bash command based on the description provided by the user.
Your output must be strictly the command itself, no explanations or additional information are allowed—only the command.
"""

# Initialize the OpenAI client with the API key, project, and organization
client = OpenAI(api_key=os.environ['API_KEY'])

n_count = 5
# status = client.batches.retrieve("batch_evROwUP7GFIHP47Ez7dww5Xr")
# client.batches.cancel("batch_evROwUP7GFIHP47Ez7dww5Xr")
#
# print(status)
# quit()

def batch_inference(dataset, start=0, stop=-1):
    """
    {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    """
    JSONL_PATH = './nl2cmd_batch.jsonl'
    entries = []
    if stop < 0:
        stop = len(dataset)

    def to_jsonline(entry_):
        return json.dumps(entry_)

    def dump_jsonl(jsonl):
        with open(JSONL_PATH, 'w') as f:
            for index, line in enumerate(jsonl):
                f.write(line + ('\n' if index < len(jsonl) - 1 else ''))

    current = 0
    for key, invocation_cmd in dataset.items():
        if current < start:
            continue
        current += 1
        start += 1
        prompt = (together_api.base_prompt +
                  together_api.prompt_format.format(invocation_cmd['invocation']))
        entry = {
            'custom_id': key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                'model':"gpt-4o-mini",
                'messages':[
                    {"role": "system", "content": CHAT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                'logprobs':True,
                'max_tokens':128,
                'n':n_count,
                'temperature':1.25, # see https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
            }
        }
        entry = to_jsonline(entry)
        entries.append(entry)
        print(entry)
        if stop > 1000:
            break
        stop += 1
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
    batch_inference(together_api.dataset, start=1002, stop=3000)
    # Print the response from the model
    # print(completion.to_json())
