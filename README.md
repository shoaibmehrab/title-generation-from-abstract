# Scientific Title Generation from Abstracts: A Comparative Study of Fine-Tuned Seq2Seq and Modern LLM Architectures

This repository trains and evaluates sequence-to-sequence models that generate scientific paper titles from abstracts.

Main models:
- FLAN-T5 Base
- BART Large CNN
- PEGASUS XSum
- PEGASUS PubMed

The script pipeline is the source of truth for reproducible experiments. Notebook files are references only.

## 1. Project Layout

```text
project/
  configs/
    base.yaml
    models/
      flan_t5_base.yaml
      bart_large_cnn.yaml
      pegasus_xsum.yaml
      pegasus_pubmed.yaml
      gemma2_9b_ollama.yaml
      llama3_8b_ollama.yaml
      llama31_8b_ollama.yaml
    training/
      quick.yaml
      full.yaml
  scripts/
    prepare_data.py
    train_seq2seq.py
  src/titlegen/
    training
    llm 
    config.py
    data/dataset.py
    training/metrics.py
    training/runtime.py
  outputs/
```

Important files:
- `configs/base.yaml`: data path, split settings, output root, global training defaults.
- `scripts/prepare_data.py`: creates deterministic train/val/test splits.
- `scripts/train_seq2seq.py`: training, generation, and metrics export.
- `src/titlegen/data/dataset.py`: dataset cleaning, schema mapping, splitting.
- `src/titlegen/training/metrics.py`: BLEU/ROUGE/BERTScore calculation.
- `src/titlegen/llm/ollama_client.py`:
- `src/titlegen/llm/prompting.py`:
- `src/titlegen/llm/quality_metrics.py`:

## 2. Environment Setup

From project root:

```bash
cd /home/shoaib/nlp-sp26/project
conda activate nlp-sp26
pip install -r requirements.txt
```

## 3. Data Configuration

Default dataset path is configured in `configs/base.yaml`:

- `/home/shoaib/nlp-sp26/project/Data/final/scientific-article-nlp.csv`


## 4. Reproducibility Rules

For fair comparison across models:
- Keep the same split and seed for all model runs.
- Use one prepared split run (example: `prep_seed42`).
- Change only model config and run name.

Recommended fixed values:
- `project.seed=42`
- `data.split.random_state=42`

## 5. Step-by-Step Reproducible Pipeline

### Step 1: Prepare deterministic splits (run once)

```bash
python scripts/prepare_data.py \
  --base-config configs/base.yaml \
  --set project.run_name=prep_seed42 \
  --set project.seed=42 \
  --set data.split.random_state=42
```

This writes split artifacts to:
- `outputs/runs/prep_seed42/splits/`

### Step 2: Train each model using the same prepared split (We used full)

#### FLAN-T5 Base (quick)

```bash
python scripts/train_seq2seq.py \
  --base-config configs/base.yaml \
  --model-config configs/models/flan_t5_base.yaml \
  --training-config configs/training/quick.yaml \
  --prepared-splits-run-name prep_seed42 \
  --set project.run_name=flan_t5_quick_seed42
```

#### FLAN-T5 Base (full)

```bash
python scripts/train_seq2seq.py \
  --base-config configs/base.yaml \
  --model-config configs/models/flan_t5_base.yaml \
  --training-config configs/training/full.yaml \
  --prepared-splits-run-name prep_seed42 \
  --set project.run_name=flan_t5_full_seed42
```

#### BART Large CNN (full)

```bash
python scripts/train_seq2seq.py \
  --base-config configs/base.yaml \
  --model-config configs/models/bart_large_cnn.yaml \
  --training-config configs/training/full.yaml \
  --prepared-splits-run-name prep_seed42 \
  --set project.run_name=bart_large_cnn_seed42
```

#### PEGASUS XSum (full)

```bash
python scripts/train_seq2seq.py \
  --base-config configs/base.yaml \
  --model-config configs/models/pegasus_xsum.yaml \
  --training-config configs/training/full.yaml \
  --prepared-splits-run-name prep_seed42 \
  --set project.run_name=pegasus_xsum_seed42
```

#### PEGASUS PubMed (full)

```bash
python scripts/train_seq2seq.py \
  --base-config configs/base.yaml \
  --model-config configs/models/pegasus_pubmed.yaml \
  --training-config configs/training/full.yaml \
  --prepared-splits-run-name prep_seed42 \
  --set project.run_name=pegasus_pubmed_seed42
```
## 6. LLM Baselines (Ollama)

In addition to seq2seq models, this project evaluates local LLMs through Ollama using:

* Llama 3.1 8B
* Llama 3 8B
* Gemma 2 9B

### Relevant Files

* **LLM eval script:** `scripts/eval_ollama_titles.py`
* **Model config (Llama 3.1):** `configs/models/llama31_8b_ollama.yaml`
* **Model config (Llama 3):** `configs/models/llama3_8b_ollama.yaml`
* **Model config (Gemma 2):** `configs/models/gemma2_9b_ollama.yaml`

---

### LLM Run Command Pattern

```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config <llm_model_config.yaml> \
  --prepared-splits-run-name prep_seed42 \
  --eval-split <val|test|both> \
  --set project.run_name=<your_run_name> \
  --set model.num_shots=<0_or_4>
```

---

### Examples

**Llama 3.1 8B (zero-shot, val)**
```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config configs/models/llama31_8b_ollama.yaml \
  --prepared-splits-run-name prep_seed42 \
  --eval-split val \
  --set project.run_name=llama31_8b_zero_val_seed42 \
  --set model.num_shots=0
```

**Llama 3.1 8B (few-shot, 4-shot, val)**
```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config configs/models/llama31_8b_ollama.yaml \
  --prepared-splits-run-name prep_seed42 \
  --eval-split val \
  --set project.run_name=llama31_8b_few4_val_seed42 \
  --set model.num_shots=4
```

**Llama 3 8B (zero-shot, val)**
```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config configs/models/llama3_8b_ollama.yaml \
  --prepared-splits-run-name prep_seed42 \
  --eval-split val \
  --set project.run_name=llama3_8b_zero_val_seed42 \
  --set model.num_shots=0
```

**Llama 3 8B (few-shot, 4-shot, val)**
```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config configs/models/llama3_8b_ollama.yaml \
  --prepared-splits-run-name prep_seed42 \
  --eval-split val \
  --set project.run_name=llama3_8b_few4_val_seed42 \
  --set model.num_shots=4
```

**Gemma 2 9B (zero-shot, val)**
```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config configs/models/gemma2_9b_ollama.yaml \
  --prepared-splits-run-name prep_seed42 \
  --eval-split val \
  --set project.run_name=gemma2_9b_zero_val_seed42 \
  --set model.num_shots=0
```

**Gemma 2 9B (few-shot, 4-shot, val)**
```bash
python scripts/eval_ollama_titles.py \
  --base-config configs/base.yaml \
  --model-config configs/models/gemma2_9b_ollama.yaml \
  --prepared-splits-run-name prep_seed42 \
  --eval-split val \
  --set project.run_name=gemma2_9b_few4_val_seed42 \
  --set model.num_shots=4
```

---

### LLM Output Artifacts

Each LLM run is saved under `outputs/runs/<run_name>/` with the following:

* `resolved_config.yaml`
* `metrics.json`
* `predictions_val.csv` or `predictions_test.csv`
* `few_shot_examples.csv` (when `num_shots > 0`)
* `human_eval/<split>_human_eval.csv`
* `human_eval/<split>_human_eval_sample_100.csv`


### Other run directory is created under:

Each Seq2seq model run is saved under `outputs/runs/<run_name>/`

Typical artifacts:
- `resolved_config.yaml`
- `trainer/` (trainer checkpoints/logs)
- `best_model/` (best saved model + tokenizer)
- `predictions_val.csv`
- `predictions_test.csv`
- `metrics.json`

The prepared split run also contains:
- `splits/train.csv`
- `splits/val.csv`
- `splits/test.csv`

---


## 7. Metrics for Reporting

Primary metrics for model comparison:
- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore F1

Reported test metrics from each run's `metrics.json`.

**Additional LLM diagnostics:**
* `exact_match_rate`
* `semantic_cosine_mean`



## 8. Known Practical Notes

- If GPU memory is tight, reduce `per_device_train_batch_size` and `per_device_eval_batch_size`, and increase `gradient_accumulation_steps`.
- You may see a warning when both `max_new_tokens` and `max_length` are set; generation still runs and `max_new_tokens` takes precedence.

