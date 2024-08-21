import json
import math
import os
from copy import deepcopy
from time import sleep

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets import DatasetDict
from transformers.models.auto.image_processing_auto import model_type
from unsloth import FastLanguageModel

from evaluate import compute_score

num_return_sequences = 5
max_seq_length = 1024
dtype = None
load_in_4bit = True

access_token = os.getenv("HF_ACCESS_TOKEN")

models = [
    {
        "model_name": "unsloth/codegemma-2b",
        "model_type": "completion",
        "batch_size": 5,
    }
]

jsonl_datasets = [
    "./data/nl2bash-data.json",
    # "./data/magnum_chatGPT_generated_data.json",
]

banned_gemma_tokens = [
    "<code>",
    "</code>",
]

temperature = 0.6

COMMAND_PROMPT = """Command Description: {}
Bash Command: """


CHAT_PROMPT = """You are a professional bash command writer.
Your sole task is to write a bash command based on the description provided by the user.

Command Description: {}

Your output must be strictly the command itself. 
No explanations or additional information are allowed—only the command."""


def generate_with_logprobs(model, tokenizer, model_type, input_texts, max_new_tokens=128, stop_token="\n") -> list[list[dict]]:
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
                ).to("cuda")
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_k=50,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    eos_token_id=tokenizer.encode(stop_token)[-1],
                )
                input_length = inputs.size(1)
                generated_tokens = outputs.sequences[:, input_length:]
            case "completion":
                inputs = tokenizer([COMMAND_PROMPT.format(text) for text in input_texts], return_tensors="pt",
                                   padding=True).to("cuda")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_k=50,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    eos_token_id=tokenizer.encode(stop_token)[-1],
                )
                generated_tokens = outputs.sequences[:, inputs['input_ids'].size(-1):]
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

        batch_size = len(input_texts)
        total_generated_sequences = batch_size * num_return_sequences
        logits = torch.stack(outputs.scores, dim=1)
        logits = logits.view(total_generated_sequences, -1, logits.size(-1))
        log_probs = F.log_softmax(logits, dim=-1)

        if generated_tokens.size(1) > 0:
            log_probs_for_generated_tokens = log_probs.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)

            for i in range(batch_size):
                batch_results = []
                for seq_idx in range(num_return_sequences):
                    sequence_result = []
                    sequence_index = i * num_return_sequences + seq_idx
                    tokens = generated_tokens[sequence_index]
                    generated_text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
                    for j, token in enumerate(tokens):
                        token_str = tokenizer.decode([token.item()], skip_special_tokens=True)
                        if token_str in banned_gemma_tokens:
                            continue
                        logprob = log_probs_for_generated_tokens[sequence_index, j].item()
                        sequence_result.append({"token_str": token_str, "log_prob": logprob})

                        if stop_token in token_str:
                            break
                    batch_results.append({"tokens": sequence_result, "text": generated_text})
                results.append(batch_results)
        else:
            results.append([{"tokens": [], "text": ""} for _ in range(num_return_sequences)])
    except Exception as e:
        print(f"Error: {e}")
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
            token=access_token
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
            output_file = f"{dir_name}/zero-shot/{model_name.split('/')[1]}.jsonl"
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
                        ground_truth_cmd = nl["ground_truth_cmd"]
                        batch_results = {
                            "index": nl["index"],
                            "predictions": [],
                            "confidences": [],
                            "global_score": 0.0,
                            "ground_truth": ground_truth_cmd
                        }
                        batch_results_fixed = batch_results.copy()
                        for sequence in result:
                            predicted_confidence = sum([math.exp(token["log_prob"]) for token in sequence["tokens"]]) / len(sequence["tokens"])
                            predicted_cmd = sequence["text"].split("\n")[0].replace("<code>", "").replace("</code>","").strip()
                            batch_results["predictions"].append(predicted_cmd)
                            batch_results["confidences"].append(predicted_confidence)
                            if "find." in predicted_cmd:
                                fixed_command = predicted_cmd.replace("find.", "find .")
                                batch_results_fixed["predictions"].append(fixed_command)
                                batch_results_fixed["confidences"].append(predicted_confidence)
                        batch_results["scores"] = compute_score(
                            ground_truths=[ground_truth_cmd],
                            predicted_cmds=batch_results["predictions"],
                            predicted_confds=batch_results["confidences"],
                            metric_params={"u1":1.0, "u2":1.0}
                        )
                        if len(batch_results_fixed["predictions"]) > 0:
                            with open(f"{dir_name}/zero-shot/{model_name.split('/')[1]}_find-dot-fix.jsonl", "a") as f_fix:
                                batch_results_fixed["scores"] = compute_score(
                                    ground_truths=[ground_truth_cmd],
                                    predicted_cmds=batch_results_fixed["predictions"],
                                    predicted_confds=batch_results_fixed["confidences"],
                                    metric_params={"u1":1.0, "u2":1.0}
                                )
                                f_fix.write(json.dumps(batch_results_fixed) + "\n")
                        f.write(json.dumps(batch_results) + "\n")

        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        sleep(5)

def train():
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 128 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # TRAIN_COMMAND_PROMPT = """Description: {}.\nBash Command: {} """
    # VALIDATION_COMMAND_PROMPT = """Description: {}.\nBash Command: """

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["invocation"]
        inputs       = examples["cmd"]
        texts = []
        for instruction, input in zip(instructions, inputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = COMMAND_PROMPT.format(instruction, input) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    from datasets import load_from_disk, load_dataset

    dataset = load_dataset('AnishJoshi/nl2bash-custom')
    dataset = dataset.remove_columns('srno')
    dataset = dataset.rename_column('bash_code', 'invocation')
    dataset = dataset.rename_column('nl_command', 'cmd')


    validation_dataset = dataset['validation']
    train_dataset = dataset['train']
    train_dataset = train_dataset[:1]
    test_dataset = dataset['test']


    from neurips2020utils.metric.metric_utils import compute_metric
    # dataset
    from transformers import TrainingArguments, TrainerCallback, DefaultFlowCallback, TrainerState, TrainerControl


    class CustomCallback(DefaultFlowCallback):

        def __init__(self, trainer, model, tokenizer, validation_dataset: DatasetDict) -> None:
            super().__init__()
            self._trainer: SFTTrainer = trainer
            self._epochs = 0
            self._validation_dataset: DatasetDict = validation_dataset
            self.batches_generation = 5
            self._model = model
            self._tokenizer = tokenizer

        def on_epoch_end(self, args, state, control, **kwargs):
            super().on_epoch_end(args, state, control, **kwargs)
            self._epochs += 1
            if self._epochs % 1 == 0:
                btc = 0
                results = []
                for entry in self._validation_dataset:
                    invocations = []
                    commands = []
                    for i in range(self.batches_generation):
                        invocation = entry["invocation"]
                        command = entry["cmd"]

                        invocations.append(invocation)
                        commands.append(command)
                        btc += 1
                    results_batch = generate_with_logprobs(self._model, self._tokenizer, model_type='completion', input_texts=invocations)
                    pass




        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_evaluate(args, state, control, **kwargs)
            pass




    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     predicted_cmds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #     log_probs = F.log_softmax(logits, dim=-1)
    #     probs = np.exp(log_probs)
    #     confidence = np.sum(probs) / len(probs)
    #     # Assuming your task is a classification task and labels are already encoded properly.
    #     return {"accuracy": compute_metric(labels, confidence, predictions)}



    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # test_dataset=test_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,

        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=10,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs = 2, # Set this for 1 full training run.
            # max_steps=None,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
        # compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer, model, tokenizer, validation_dataset))
    trainer_stats = trainer.train()

if __name__ == "__main__":
    # run()
    train()
