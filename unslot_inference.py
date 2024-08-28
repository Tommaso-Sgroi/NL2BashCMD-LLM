import json
import math
import os
from time import sleep

import torch
import torch.nn.functional as F
import tqdm
# from plyer import temperature
from unsloth import FastLanguageModel

from neurips2020utils.metric.metric_utils import compute_metric

max_seq_length = 1024
dtype = None
load_in_4bit = False
temperature = 0.2
models = [
    {
        "model_name": "unsloth/codellama-7b",
        "model_type": "completion",
        "batch_size": 10,
    }
]

jsonl_datasets = [
    "./data/nl2bash-data.json",
    "./data/magnum_chatGPT_generated_data.json",
]

banned_gemma_tokens = [
    "<code>",
    "</code>",
]

COMMAND_PROMPT = """Task: Convert the following descriptions into bash commands.
Command Description: {}
Bash Command: """


CHAT_PROMPT = """You are a professional bash command writer.
Your sole task is to write a bash command based on the description provided by the user.

Command Description: {}

Your output must be strictly the command itself. 
No explanations or additional information are allowed—only the command."""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_with_logprobs(model, tokenizer, model_type, input_texts, max_new_tokens=128, stop_token="\n") -> list[dict]:
    results = []
    try:
        match model_type:
            case "chat":
                messages_batch = []
                for input_text in input_texts:
                    messages = [
                        {"role": "user", "content": CHAT_PROMPT.format(input_text)},
                    ]
                    messages_batch.append(messages)
                inputs = tokenizer.apply_chat_template(
                    messages_batch,
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=tokenizer.encode(stop_token)[-1],
                )
                input_length = inputs.size(1)
                generated_tokens = outputs.sequences[:, input_length:]
            case "completion":
                inputs = tokenizer([COMMAND_PROMPT.format(text) for text in input_texts], return_tensors="pt",
                                   padding=True).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=tokenizer.encode(stop_token)[-1],
                )
                generated_tokens = outputs.sequences[:, inputs['input_ids'].size(-1):]
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

        logits = torch.stack(outputs.scores, dim=1)
        log_probs = F.log_softmax(logits, dim=-1)

        if generated_tokens.size(1) > 0:
            log_probs_for_generated_tokens = log_probs.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)

            for i, tokens in enumerate(generated_tokens):
                sequence_result = []
                generated_text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
                for j, token in enumerate(tokens):
                    token_str = tokenizer.decode([token.item()], skip_special_tokens=True)
                    if token_str in banned_gemma_tokens:
                        continue
                    logprob = log_probs_for_generated_tokens[i, j].item()
                    sequence_result.append({"token_str": token_str, "log_prob": logprob})

                    if stop_token in token_str:
                        break
                results.append({"tokens": sequence_result, "text": generated_text})
        else:
            results.append({"tokens": [], "text": ""})
    except Exception as e:
        print(f"Error: {e}")
        pass
    return results


def load_dataset_from_file(file_path):
    entries = []
    with open(file_path, "r") as f:
        if 'jsonl' not in f.name and 'json' in f.name:
            tmp_entries = json.load(f)
            for k,v in tmp_entries.items():
                v['index'] = k
                entries.append(v)
        else:
            for line in f:
                entries.append(json.loads(line))
    return entries


def load_processed_indexes(file_path):
    indexes = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                indexes.append(json.loads(line)["index"])
    return indexes


def run():
    for model_info in models:
        model_name = model_info["model_name"]
        batch_size = model_info["batch_size"]
        model_type = model_info["model_type"]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        FastLanguageModel.for_inference(model)
        for dataset in jsonl_datasets:
            nls = []
            entries = load_dataset_from_file(dataset)
            nls.extend(
                [{"index": entry["index"], "invocation": entry["invocation"], "ground_truth_cmd": entry["cmd"]} for
                 entry in entries])
            nls = [nls[i:i + batch_size] for i in range(0, len(nls), batch_size)]
            dir_name = f"./benchmarks/{dataset.split('/')[-1].split('.')[0]}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            output_file = f"{dir_name}/{model_name.split('/')[1]}.jsonl"
            processed_indexes = load_processed_indexes(output_file)
            with open(output_file, "a+") as f:
                for nl_chunk in tqdm.tqdm(nls):
                    nl_chunk = [nl for nl in nl_chunk if nl["index"] not in processed_indexes]
                    if len(nl_chunk) == 0:
                        continue
                    results = generate_with_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        model_type=model_type,
                        input_texts=[nl["invocation"] for nl in nl_chunk],
                    )
                    for nl, result in zip(nl_chunk, results):
                        num_tokens = len(result["tokens"])
                        predicted_confidence = sum(
                            [math.exp(token["log_prob"]) for token in result["tokens"]]) / num_tokens
                        predicted_cmd = result["text"].split("\n")[0].replace("<code>", "").replace("</code>","").strip()
                        if "find." in predicted_cmd:
                            with open(f"{dir_name}/{model_name.split('/')[1]}_find-dot-fix.jsonl", "a") as f_fix:
                                fixed_score = compute_metric(
                                    predicted_cmd=predicted_cmd.replace("find.", "find ."),
                                    predicted_confidence=predicted_confidence,
                                    ground_truth_cmd=nl["ground_truth_cmd"]
                                )
                                json_data_fix = {"index": nl["index"],
                                                 "prediction": predicted_cmd.replace("find.", "find ."),
                                                 "confidence": predicted_confidence, "score": fixed_score,
                                                 "ground_truth": nl["ground_truth_cmd"]}
                                f_fix.write(json.dumps(json_data_fix) + "\n")
                        score = compute_metric(
                            predicted_cmd=predicted_cmd,
                            predicted_confidence=predicted_confidence,
                            ground_truth_cmd=nl["ground_truth_cmd"]
                        )
                        json_data = {"index": nl["index"], "prediction": predicted_cmd,
                                     "confidence": predicted_confidence, "score": score,
                                     "ground_truth": nl["ground_truth_cmd"]}
                        f.write(json.dumps(json_data) + "\n")
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        sleep(5)


if __name__ == "__main__":
    run()
    # load_dataset_from_file(jsonl_datasets[0])