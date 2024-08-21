"""
todo: use https://github.com/unslothai/unsloth to do inference and finetune of LLMs
maybe use https://github.com/ollama/ollama-python just for inference
maybe use https://github.com/abetlen/llama-cpp-python to use a very fast version of llama
"""

from neurips2020utils.metric.metric_utils import compute_metric

### this is a first test
gtnl = 'Set permission of "file" to read only for the owner'
gtcmd = 'find $HOME -name ".*" -ls'

predcmd = 'find path -name ".*" -ls'

score = compute_metric(predicted_cmd=predcmd, predicted_confidence=0.84135, ground_truth_cmd=gtcmd)
print(score)

