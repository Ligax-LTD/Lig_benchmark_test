# Ligax AI Hairstyle Classification Benchmark

A public benchmark evaluating Lig Vision, the hairstyle classification model powering the Ligax AI platform, against three frontier vision-language models on a labelled dataset of Afro-Caribbean protective hairstyles.

All code, pre-computed results, and the dataset are publicly available. Results for Lig Vision can be verified image by image via the Ligax AI playground at https://www.ligaxai.com/playground.

---

## Results

Evaluated on 314 images. Named style F1 computed on 283 images; base technique F1 on 278 images.

Lig Vision F1 scores are out-of-fold estimates from 5-fold cross-validated threshold calibration. Frontier model F1 scores use the models' native binary outputs without post-hoc calibration.

| Model | Named Styles F1 | Named Styles mAP | Base Technique F1 |
|---|---|---|---|
| Lig Vision | **0.622** | **0.770** | **0.780** |
| Claude Opus 4.8 | 0.477 | n/a | 0.693 |
| Gemini 2.5 Flash | 0.433 | n/a | 0.667 |
| GPT-5.5 | 0.411 | n/a | 0.572 |

mAP is not reported for frontier models as they do not return probability scores.

---

## Repository Structure

```
benchmark_test/
├── evaluate.py          main evaluation script
├── calibrate.py         5-fold CV threshold calibration for Lig Vision
├── dataset.py           fetches dataset from Roboflow (no local download)
├── metrics.py           F1 and mAP computation
├── taxonomy.json        benchmark label taxonomy
├── requirements.txt     Python dependencies
├── models/
│   ├── lig_vision.py    Lig Vision API client
│   ├── gpt.py           OpenAI GPT classifier
│   ├── gemini.py        Google Gemini classifier
│   ├── claude_model.py  Anthropic Claude classifier
│   └── _prompt.py       shared classification prompt
├── results/
│   ├── lig_vision.json  pre-computed Lig Vision results
│   ├── claude.json      Claude results
│   ├── gemini.json      Gemini results
│   └── gpt.json         GPT results
└── notebook/
    └── benchmark_results.ipynb  results visualisation
```

---

## Setup

**Requirements:** Python 3.10+

```
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```
# Roboflow — public benchmark dataset
RF_API_KEY=<your Roboflow API key>
RF_WORKSPACE=testing-workspace-htshl
RF_PROJECT=woman-hairstyles-bevyz
RF_VERSION=1

# Frontier model APIs
OPENAI_API_KEY=<your OpenAI API key>
OPENAI_MODEL=gpt-5.5

ANTHROPIC_API_KEY=<your Anthropic API key>
ANTHROPIC_MODEL=claude-opus-4-8

GOOGLE_API_KEY=<your Google API key>
GOOGLE_MODEL=gemini-2.5-flash
```

The Roboflow API key for this dataset is available from the dataset page at https://app.roboflow.com/testing-workspace-htshl/woman-hairstyles-bevyz/browse.

`LIG_VISION_URL` is intentionally omitted. Lig Vision results are pre-computed in `results/lig_vision.json`. The inference endpoint is not published; individual predictions can be verified at https://www.ligaxai.com/playground.

---

## Running the Benchmark

Evaluate all frontier models:

```
python evaluate.py
```

Evaluate a specific model:

```
python evaluate.py --models claude
python evaluate.py --models gemini
python evaluate.py --models gpt
```

Models with an existing complete result file are skipped automatically.

After running Lig Vision evaluation, run threshold calibration to update `results/lig_vision.json` with out-of-fold metrics:

```
python calibrate.py
```

---

## Verifying Lig Vision Results

The Ligax AI playground at https://www.ligaxai.com/playground uses the same model and inference endpoint that produced the benchmark results.

To verify a specific prediction:

1. Open the dataset at https://app.roboflow.com/testing-workspace-htshl/woman-hairstyles-bevyz/browse
2. Download any image from the test split
3. Upload it to the playground
4. Compare the returned labels with the corresponding entry in `results/lig_vision.json`

---

## Dataset

The benchmark dataset consists of 314 labelled images of Afro-Caribbean protective hairstyles. All images are in the test split with no overlap with Lig Vision's training data. Labels were applied and independently verified by professional hairstylists.

The dataset is fetched live from Roboflow at evaluation time. No local download is required or performed, ensuring no data leakage into the evaluation process.

Dataset: https://app.roboflow.com/testing-workspace-htshl/woman-hairstyles-bevyz/browse

---

## Taxonomy

The benchmark evaluates two label dimensions:

**Named Styles (18 classes):** alicia_keys, boho, box_braids, butterfly_locs, faux_locs, flat_twist, fulani_braids, invisible_locs, island_twist, lemonade_braids, marley_twists, microlocs, mini_twists, passion_twists, ponytails, puff, soft_locs, space_buns

**Base Technique Categories (5 classes):** braids, cornrows, crochet, locs, twist

The full taxonomy including technique modifiers, size attributes, and add-ons is defined in `taxonomy.json` and described in the benchmark documentation.

---

## Threshold Calibration

Lig Vision returns continuous probability scores. Per-class decision thresholds are calibrated using 5-fold cross-validation: for each fold, thresholds are optimised on the remaining four folds; the held-out fold is evaluated with those thresholds. No image appears in its own calibration set. The reported F1 scores are out-of-fold estimates.

This approach avoids test-set contamination from threshold optimisation. The calibration script and all intermediate outputs are included in the repository.

---

## Limitations

- Several named style classes have fewer than five ground truth examples. Per-class F1 for these classes has high variance and should be interpreted with caution.
- The `microlocs` class has no effective ground truth in the current dataset due to a column naming mismatch (`micro_locs` in the dataset vs `microlocs` in the taxonomy). It is excluded from macro F1 computation and will be resolved in the next dataset version.
- Frontier models return binary labels without probability scores. Their decision thresholds are implicit and cannot be calibrated post-hoc.
- This benchmark covers one style domain. Lig Vision operates across a broader range of hairstyle categories and demographic contexts; these results are not representative of its full capability.

---

## Contact

admin@ligaxai.com
