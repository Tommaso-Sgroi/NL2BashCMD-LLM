import os
from openai import OpenAI

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
        temperature=1, # see https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature

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
    base_prompt = "Write me a one line bash command for the following tasks, "

    together_api.dataset = get_dataset(dataset_path)
    benchmark(model_name='gpt-4o', base_prompt=base_prompt, early_stop=10)
    # Print the response from the model
    # print(completion.to_json())
