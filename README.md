# Natural Language to Bash Command Translation

[Full "Paper"](https://github.com/Tommaso-Sgroi/NL2BashCMD-LLM/blob/main/NLP_23_24___NL2BASH.pdf)

## Overview

This repository presents research on **Natural Language (NL) to Bash command translation**, an essential task in **Natural Language Processing (NLP)**. The ability to convert natural language instructions into executable Bash commands has significant applications in **automated system administration, software development, cybersecurity, and command-line education**.

The **primary goal** of this research is to evaluate **state-of-the-art Large Language Models (LLMs)** and compare their performance against traditional transformer-based models in **NL2Bash** and **NL2CMD**, two key datasets in this domain.

## LLM Responses
Under ```benchmarks``` repository, you can find the responses of the LLMs to the NL2Bash and NL2CMD datasets. The responses are stored in the ```benchmarks/nl2bash_magnum``` and ```benchmarks/nl2bash_tellina``` folders, where each subfolder contains the responses of a specific LLM to the NL2Bash and NL2CMD datasets in the format: ```[model-name]-additional_info.jsonl```. For exception of ChatGPT which responses are stored under ```benchmarks``` folder.<br/>
If nothing is specified, the all parameters are default. The info could, also, contain the dataset name.<br/>

**Feel free to use these responses for your research or any other purpose—but don’t forget to [cite us](https://github.com/Tommaso-Sgroi/NL2BashCMD-LLM#citation)!** 😉 <br/>
<sup><sub>(Because we spent real money on this and we're just broke students—at least let us have some credit and satisfaction 🫡😁)</sub></sup>

## Research Objectives

- **Benchmarking LLMs**: Compare the latest transformer-based models (e.g., GPT-4, LLaMA, ...) on the NL2Bash and NL2CMD datasets.
- **Generalization Analysis**: Evaluate models under **zero-shot** settings.

## Background & Challenges

### Why is NL to Bash Translation Difficult?

1. **Lexical Ambiguity** – Natural language is highly variable, and the same command can be phrased differently.
2. **Syntax Complexity** – Bash syntax is strict, requiring precise formatting, special characters, and argument ordering.
3. **Data Scarcity** – There are **few large-scale datasets** available for training models on this task.

### Existing Work

The **NL2Bash dataset**~[Lin et al., 2018] was one of the first major datasets in this space. Early research used **sequence-to-sequence transformers**, but models struggled to surpass **50% accuracy** due to limited training data and strict Bash syntax constraints.

A more recent dataset, **NL2CMD**~[Fu et al., 2023], extends NL2Bash by **introducing a sysntetic dataset**, improving dataset diversity. However, most models still struggle with **generalization** and **out-of-distribution commands**.

### Breakthrough: LLMs in 2023

In **2023, ChatGPT (GPT-4) achieved 80.6% accuracy** on NL2Bash in zero-shot conditions, demonstrating **LLMs' remarkable ability to infer command structures without explicit training**. However, challenges remain:

- **High computational costs** limit the practicality of large models.
- **Lack of domain adaptation** results in errors in complex Bash scripts.
- **Hallucination issues** lead to incorrect or potentially harmful commands.

## Methodology

Our research follows a **systematic evaluation framework**:

- **Model Comparisons**:
  - State-of-the-art **LLMs** (GPT-4, LLaMA, ...)
- **Inference Strategies**:
  - **Zero-shot** (pretrained models tested directly)
  - **Top five responses** (for each model generate five responses)
- **Metrics**:
  - **NeurIPS 2020 nlc2cmd competition** – Measures how often the generated command matches the reference.

## Evaluation Metrics

To assess the quality of predicted Bash commands compared to reference commands, we use the following metrics:

### **FLAG Score**
The **FLAG Score** measures the similarity between the predicted command's flags and the reference command's flags:

$$
S_{F}^{i} (F_{\text{pred}}, F_{\text{ref}}) = \frac{1}{N} \left( 2 \times |F_{\text{pred}} \cap F_{\text{ref}}| - |F_{\text{pred}} \cup F_{\text{ref}}| \right)
$$

where:
- $F_{\text{pred}}$ represents the set of predicted flags,
- $F_{\text{ref}}$ represents the set of reference flags,
- $N$ is the total number of flags.

### **Utility Score**
The **Utility Score** measures how well the predicted command matches the reference command, considering both the flags and the utility of the command itself:

$$
S_{U} = \sum_{i \in [1, T]} \frac{1}{T} \times \left( |U_{\text{pred}} = U_{\text{ref}}| \times \frac{1}{2} (1 + S_{F}^{i} ) - |U_{\text{pred}} \not= U_{\text{ref}} | \right)
$$

where:
- $U_{\text{pred}}$ is the predicted utility (command name),
- $U_{\text{ref}}$ is the reference utility,
- $S_{F}^{i}$ is the FLAG Score for the i-th instance.

### **Total Accuracy Score**
The **Total Accuracy Score** evaluates the highest possible similarity score for a given natural language command (nlc) and its generated predictions:

$$
Score(A(nlc)) =
\begin{cases} 
\max\limits_{p \in A(nlc)} S(p), & \text{if } \exists_{p \in A(nlc)} \text{ such that } S(p) > 0, \\
\frac{1}{|A(nlc)|} \sum\limits_{p \in A(nlc)} S(p), & \text{otherwise}.
\end{cases}
$$

where:
- $A(nlc)$ represents the set of all generated predictions for a given natural language command.
- $S(p)$ is the score function that measures the accuracy of a single predicted command.

This metric ensures that **if at least one generated command is correct**, the best score is used. Otherwise, the **average similarity score across all predictions** is considered.

These metrics provide a structured way to evaluate model performance, ensuring that both the correctness of command utilities and the accuracy of command flags are considered.

## Results

The results were **disappointing**, as even **ChatGPT-4o struggled to reach an overall score of 0.2**. In contrast, **Magnum Research Group using ChatGPT achieved 0.80**, raising concerns about potential leaks in older versions of ChatGPT.

One major challenge faced by LLMs was the **evaluation metric itself**. While LLMs were generating **valid Bash commands**, they often used **different strategies** (from the ground truth) to achieve the same goal. This suggests that the metric may have **biased the competition towards dataset-specific solutions**, leading models to **overfit to preferred command structures** rather than focusing on general command validity.

These findings highlight a key limitation in current benchmarks: models that explore diverse solutions may be penalized despite producing correct outputs. Further analysis is needed to refine evaluation methodologies to ensure fair and meaningful comparisons.

## Contributors
[Tommaso Sgroi](https://github.com/Tommaso-Sgroi),
[Damiano Gualandri](https://github.com/dag7dev),
[Luca Scutigliani](https://github.com/scuty2000)

## Citation
If you are using our LLMs generated responses —**or any part of our work for that matter**— we’d really appreciate it if you could at least give us some credit. 
```
@misc{nl2bashcmd2024, 
  author = {Tommaso Sgroi, Damiano Gualandri, Luca Scutigliani}, 
  title = {NL2BashCMD: State of The Art Evaluation of Natural Language to Bash Commands Translation}, 
  year = {2024},
  howpublished = {\url{https://github.com/Tommaso-Sgroi/NL2BashCMD-LLM}}
  }
```

