"""
todo: use https://github.com/unslothai/unsloth to do inference and finetune of LLMs
maybe use https://github.com/ollama/ollama-python just for inference
maybe use https://github.com/abetlen/llama-cpp-python to use a very fast version of llama
"""
from pyexpat.errors import messages

# from neurips2020utils.metric.metric_utils import compute_metric
# from utils.ollama import *
# from utils.data import get_dataset
from  llama_cpp import Llama
import torch
### this is a first test
# gtnl = 'Set permission of "file" to read only for the owner'
# gtcmd = 'find $HOME -name ".*" -ls'
#
# predcmd = 'find path -name ".*" -ls'
#
# score = compute_metric(predicted_cmd=predcmd, predicted_confidence=0.84135, ground_truth_cmd=gtcmd)
# print(score)


# ds = get_dataset('./data/nl2bash-data.json')

# llm = Llama.from_pretrained(
#     repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
#     filename="*Q2_K.gguf",
#     verbose=True,
#       main_gpu=0,
#       logits_all=True,
# )
#
# output = llm(
#       prompt='say hello!',
#       logprobs=True,
#       temperature=0,
#
# )
import math

# Example response from your server (shortened for clarity)
response = {
    "choices": [{
        "logprobs": {
            "token_logprobs": [-2.388612747192383, -0.022651387378573418, -0.022099770605564117, -0.1866966336965561, -1.2244893312454224, -0.1470845490694046, -0.6026426553726196, -0.12992003560066223, -1.1403454542160034, -1.9360718727111816, -0.0011967408936470747, -0.17048537731170654, -0.003275032388046384, -0.9937889575958252, -0.022069688886404037, -0.18474289774894714],
            "tokens": [" Step", " ", "1", ":", " Choose", " a", " Domain", " Name", "\n", "Step", " ", "2", ":", " Select", " a", " Web"]
        }
    }]
}

# Extract tokens and log probabilities
logprobs = response['choices'][0]['logprobs']['token_logprobs']
tokens = response['choices'][0]['logprobs']['tokens']

# Calculate confidence (probability) for each token
confidences = [math.exp(logprob) for logprob in logprobs]

# Print tokens with their confidence values
for token, confidence in zip(tokens, confidences):
    print(f"Token: {token}, Confidence: {confidence:.4f}")


quit()

llm = Llama(
      model_path="./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      seed=1337, # Uncomment to set a specific seed
      main_gpu=0,
      logits_all=True,
      # n_ctx=2048, # Uncomment to increase the context window
)

# output = llm.create_chat_completion(
#       messages=[
#             {"role": "system", "content": "You are a normal person."},
#             {
#                   "role": "user",
#                   "content": [
#                         {"type": "text", "text": "say hello world!"},
#                   ]
#             }
#       ],
#       # stream=True,
#       max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       logprobs=2,
#       temperature=0,
# ) # Generate a completion, can also call create_completion
output = llm.create_chat_completion(
      messages=[
            {"role": "system", "content": "You are a normal person."},
            {
                  "role": "user",
                  "content": [
                        {"type": "text", "text": "say hello world!"},
                  ]
            }
      ],
      # stream=True,
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      logprobs=True,
      stream=True,
      temperature=0,
) # Generate a completion, can also call create_completion

for chunk in output:
      print(chunk)

print(output)
quit()
results = output['choices'][0]['text']
print('Results from the model:\n')
print(results)

log_probs = output['choices'][0]['logprobs']['top_logprobs']
print('Results by selecting tokens with the highest probabilities:\n')
for el in log_probs:
    chosen = max(el, key=lambda k: el[k])
    print(chosen, end = '')