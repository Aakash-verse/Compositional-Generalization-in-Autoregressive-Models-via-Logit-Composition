# Supplementary Code

This anonymous supplement contains the notebook used for the synthetic letter-replacement experiment and the logit-composition LLM evaluations.

## Environment

- Python 3.12 was used for the submitted runs.
- Install the notebook dependencies with:

```bash
pip install torch transformers datasets lm_eval evalplus pandas tqdm
```

- The LLM evaluations require Hugging Face access to `google/gemma-2-2b` and the MergeBench checkpoints. Set the token outside the notebook:

```bash
export HF_TOKEN=<your-huggingface-token>
```

The notebook intentionally does not contain tokens, author names, local paths, or institution-specific paths.

## Models

- Base: `google/gemma-2-2b`
- Math expert: `MergeBench/gemma-2-2b_math`
- Coding expert: `MergeBench/gemma-2-2b_coding`

The merged decoder uses greedy decoding with `MAX_NEW = 512`, `SAMPLE_MERGED = False`, `torch.manual_seed(0)`, float16 on CUDA, and `device_map="auto"` when CUDA is available.

## Benchmarks

- GSM8K: `lm_eval.simple_evaluate`, task `gsm8k`, 8-shot, batch size 1, flexible exact-match score reported.
- MATH: `DigitalLearningGmbH/MATH-lighteval`, all test subjects, two-shot prompt in the notebook, boxed-answer exact match after light normalization.
- HumanEval+: `evalplus`, dataset `humaneval`, pass@1.
- MBPP+: `evalplus`, dataset `mbpp`, pass@1.

## Reported Results

| Benchmark | Base | Coding expert | Math expert | Logit composition |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 3.2 | - | 50.2 | 41.2 |
| MATH | 16.3 | - | 25.5 | 24.2 |
| HumanEval+ | 3.7 | 24.4 | 13.4 | 30.5 |
| MBPP+ | 33.9 | 38.1 | 32.8 | 39.4 |

The notebook writes generated samples for EvalPlus to `samples_humaneval.jsonl` and `samples_mbpp.jsonl`; these files are not included in the supplement.
