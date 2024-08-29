import os

import g4f
from g4f.models import gpt_4, gpt_4_turbo

import together_api
from together_api import *
from g4f import Model
from g4f.client import Client
import logging as logger

g4f_client = Client(proxies=proxies)
active_proxies = g4f_client.get_proxy()
if not active_proxies:
    logger.error("No active proxies! Check your configuration for any error.")
    continue_prompt = input("Do you want to continue anyway? (y/N)")
    if continue_prompt.lower() != "y":
        exit(0)
    print("Running without proxies is highly discouraged as your IP will be exposed.")
else:
    print(f"Active proxy configuration: {active_proxies}")



def get_model(name: str) -> str:
    """
    Get a G4F model by name.

    Args:
        name (str): The name of the model.

    Returns:
        Model: The G4F model.
    """
    models_list = g4f.Model.__all__()
    if name not in models_list:
        error_msg = f"Model {name} not found.\n Available models: {models_list}"
        raise ValueError(error_msg)

    return name

n_count = 5
def inference(prompt):
    # data_request = RequestData(**data)
    predictions = []
    for _ in range(n_count):
        response = g4f_client.chat.completions.create(
            provider=g4f.Provider.Raycast,
            # model=model,
            messages=[{"role": "user", "content": f'{prompt}'}],
            logprobs = True,
            top_logprobs = 5,
            **data
        )
        text = response.choices[0].message.content
        text = text.replace('```bash\n', '').replace('```', '').strip()
        predictions.append(text)
        # print(response)
    return predictions, [1.0] * n_count

together_api.inference = inference


# quit()


# print(inference('hello, nice to meet you.'))
if __name__ == '__main__':
    del data['logprobs']
    del data['stop']


    base_prompt = """You are a professional bash command writer.
Your sole task is to write a bash command based on the description provided by the user.

Command Description: {}

Your output must be strictly the command itself. 
No explanations or additional information are allowed—only the command."""

    together_api.prompt_format = 'Description: {} \nBash command: '

    together_api.dataset = get_dataset(dataset_path)
    model_path = os.getenv('MODEL_PATH')
    benchmark(model_name=model_path, base_prompt=base_prompt)

