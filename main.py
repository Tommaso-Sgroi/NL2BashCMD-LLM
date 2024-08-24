"""
todo: use https://github.com/unslothai/unsloth to do inference and finetune of LLMs
maybe use https://github.com/ollama/ollama-python just for inference
maybe use https://github.com/abetlen/llama-cpp-python to use a very fast version of llama
"""
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
