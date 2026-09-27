# spec-decoding

## ToDo
### Before running tests:
- [ ] Increase max samples in the config files
- [ ] Double check if [eval_kl](src/tasks/distillation/eval_kl.py) file
- [ ] Find a way to add different num heads for medusa
- [ ] Find a way to add EAGLE-1 (Static Tree) and EAGLE-2 (Dynamic Tree) for testing

### Before running spec decoding for medusa and eagle:
- [ ] Add a way to save weights for individual additions to the base model

#### For `train_heads`
- [ ] Adjust the training sequence length (`max_seq_len`), batch size, learning rate, and total training epochs/steps
- [ ] Test 8 and 16 heads

#### For `train_eagle`
- [ ] Configure the tree draft head parameters, sequence truncation length, and hidden state feature extraction setup

#### Train linear addition of base model
- [ ] Run the training for medusa and eagle

### Running Experimental Benchmarks & Evaluation
- [ ] Medusa Spec Decoding
  - [ ] Make sure to set `--max_new_tokens` to increase the generation token cutoff
  - [ ] Adjust candidate tree tree-structure choices to `num_heads`
- [ ] EAGLE Spec Decoding
  - [ ] Set `--max_new_tokens` and update `TREE_CHOICES`
    - ties to the eagle-1 and eagle-2 above

## Setup
Clone with submodules:
```bash
git clone --recursive git@github.com:michaelpginn/spec-decoding-capstone.git
```

Then, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it isn't already.

```bash
# Run a spec dec evaluation (inference only)
uv run run.py experiments/<config>.cfg --overrides key1=val1 key2=val2

# Generate logprob file for distillation
uv run scripts/generate_teacher_logprobs.py experiments/distillation/logprobs_general.cfg
# ... or logprobs_translation.cfg

# Run distillation
uv run scripts/distill.py experiments/distillation/distill
```

Our three main scripts each take an ini-style config file (`.cfg`). The source of truth for config parameters is in `src/config/config.py`.

## Research Questions

1. Do **low-resource languages** face worse speedup factors than high-resource languages?
2. For LR languages, it more effective to use draft models that are created via **knowledge distillation** or trained for **language modeling** on monolingual corpora?
    1. For KD, is it better to use a **quantized model** or **smaller model** with a similar architecture?
    2. For LM, is it better to use a **neural model** or an **n-gram model**?
3. How can we practically implement **draft model routing** for a multilingual language model?

## Links
- [📝 Notes doc](https://docs.google.com/document/d/1GcsLQniqIWbxFAj_zbTSZS0302S73-ZZPJ2WA_w1w9g/edit?usp=sharing)
- [📆 Project timeline](https://www.notion.so/Multilingual-Speculative-Decoding-2bc9f22610ac80a98c0bf2eedb6e3457?source=copy_link)
