# Group 63 — Privacy attacks on record-linkage models

COMP3850, Session 2 2026, Macquarie University.

This project evaluates whether a record-linkage model's output reveals whether a **record pair** was used during training. Linkage labels (match/non-match) and membership labels (member/non-member) are different tasks.

## Notebooks

| File | Purpose | Default behaviour |
|---|---|---|
| `SNN Final end to end (1).ipynb` | Compare three record-linkage baselines | Trains models and saves a new timestamped results folder |
| `K_Shadow_Prototype.ipynb` | Compare membership attacks using five and ten shadow models | Reads and verifies an existing saved experiment |

Run notebooks from the repository root, in cell order. The SNN notebook is not a prerequisite for reviewing the saved shadow experiment. The notebooks use separate target-training runs.

## Installation: Windows and Mac

Use **64-bit Python 3.10** in a fresh Conda environment. The single `requirements.txt` selects the TensorFlow package automatically by operating system and Python architecture. Apple Silicon requires native ARM64 Python and macOS 12 or newer. Intel Macs use the Intel TensorFlow wheel. This setup uses CPU execution; no Metal or CUDA plugin is required.

Open Anaconda Prompt on Windows, or a Conda-enabled Terminal on Mac. Change to this repository's directory first. On Windows, `cd /d "C:\path\to\Group-63-COMP3850"` also switches drives; on Mac use `cd /path/to/Group-63-COMP3850`.

```sh
conda create -n snn-env-check python=3.10 -y
conda activate snn-env-check
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --user --name snn-env-check --display-name "Python (snn-env-check)"
python -c "import tensorflow as tf, numpy, pandas, sklearn, matplotlib, seaborn, IPython, jupyter_client; print('TensorFlow:', tf.__version__); print('Imports successful')"
```

Use VS Code with Microsoft's Python and Jupyter extensions. Open the repository folder, open a notebook, select **Python (snn-env-check)** in the kernel selector, and restart the kernel before running cells in order. If it is not listed, select the Python interpreter inside that Conda environment. Confirm from a notebook cell with `import sys; print(sys.executable)`.

Keep an existing working `snn-env` until the new environment passes these checks. Installing this file into an environment containing newer JupyterLab/AnyIO packages can leave conflicts. This requirements file supplies the VS Code kernel, not a standalone JupyterLab server. Do not upgrade typing_extensions independently: this project retains TensorFlow 2.13.1 for compatibility with its saved models.

This is a pinned direct-dependency specification, not a complete transitive lockfile. Native Windows/Mac installation remains to be verified on teammates' computers. After a successful installation, record the environment with `python -m pip freeze > environment-installed.txt` and include the OS/architecture in the filename if sharing it.

## Data and large files

Obtain the team's GitHub datasets and cached embeddings. A support-file ZIP does not include them. If using a clone with Git LFS configured:

```sh
git status
git pull --ff-only origin main
git lfs pull origin
git lfs ls-files
```

Preserve local edits before pulling. Git LFS downloads only files actually published to that repository; it cannot supply missing uploads. If Git LFS is not installed, install it and run `git lfs install`; do not force-overwrite an existing hook without reviewing it.

Expected CSVs in `Datasets/`: `target_train.csv`, `target_test.csv`, `shadow_train.csv`, `shadow_test.csv`. Expected fields: `uid1`, `uid2`, `text1`, `text2`, `label`.

Expected arrays in `Embeddings/`: for each split `target_train`, `target_test`, `shadow_train`, `shadow_test`, include `x1_<split>.npy`, `x2_<split>.npy`, and `y_<split>.npy` (12 files total).

| Split | x1 and x2 shape | y shape |
|---|---|---|
| target_train / shadow_train | (32000, 768) | (32000,) |
| target_test / shadow_test | (8000, 768) | (8000,) |

Never independently sort, drop or reorder CSV rows or embedding arrays. Label agreement is checked, but is not proof of text-to-embedding correspondence. An earlier shadow-test CSV had malformed rows: use the team's corrected source; do not silently discard rows to make shapes match.

The SNN notebook loads all 12 arrays and the two target CSVs. The shadow notebook's default saved-results mode still imports TensorFlow and reads **target_train.csv and shadow_train.csv** for pair separation. New shadow training/reload verification additionally needs target/shadow training embeddings. The external test arrays are not used in the main membership experiment.

The historical BERT generation code is preserved as Markdown reference only. It is not executed by Run All. Transformers is therefore not required for the current cached-embedding workflow. Regeneration needs a separately reviewed embedding pipeline and alignment checks; do not replace the cached embeddings casually.

## Review the completed membership experiment

Keep the complete folder `k_runs/main_20260920_230533/` and its internal filenames. In the configuration cell, retain:

```python
TRAIN_NEW_RUN = False
SAVED_RUN = Path("k_runs") / "main_20260920_230533"
SHADOW_COUNTS = (5, 10)
VERIFY_MODEL_RELOAD = False
```

Run all cells. This loads saved target/shadow outputs, recalculates attack metrics, checks them against saved metrics, and audits membership indices. It leaves result files unchanged. The reload report displayed in this mode is **historical**, not a live verification.

For live model verification, set `VERIFY_MODEL_RELOAD = True` and provide the original training embeddings. This checks a sample of each neural model's outputs and all saved attack scores. Load only trusted project model/joblib files.

## Train a new membership experiment

Set `TRAIN_NEW_RUN = True`, retaining the intended five/ten-shadow configuration. Restart and run in order. A new timestamped `k_runs/main_<timestamp>/` is created; prior experiments are preserved.

Each target/shadow pool contains 32,000 pairs. Each model trains on 22,400 (70%) and holds out 9,600 (30%). It queries 6,720 training members and 6,720 held-out non-members. Members intentionally come from training; non-members must not. Each encoder trains for 30 epochs and each linkage head for 20, with batch size 256. Shadow seeds differ; target seed is 2026.

The attack uses linkage probability, maximum class probability and binary entropy with StandardScaler + LogisticRegression. It fits only on shadow outputs. Both K values are evaluated on the same fixed target queries at membership threshold 0.5. K=5 uses the first five of the ten trained shadows.

Do not change configuration/data midway through a run. Completion markers allow the shadow-training cell to skip completed shadows within the same active experiment; this is not a general restart/resume interface. Re-running target training creates a new experiment. Re-running a completed new-run attack cell fails rather than overwriting its output directory.

## Saved artifacts

| Location | Contents |
|---|---|
| `results/linkage_<timestamp>/` | Baseline comparison and separate Siamese-head/direct-MLP folders |
| `siamese_head/` | Encoder, head, metrics, test predictions, training histories, confusion matrix, distance metrics |
| `direct_embedding_mlp/` | Model, metrics, test predictions, training history, threshold search, split indices, confusion matrix |
| `k_runs/main_<timestamp>/target/` and `shadow_<n>/` | Models, membership predictions, split indices and histories |
| `k_runs/main_<timestamp>/attack_evaluation/` | Attack pipelines, target predictions, comparison table and ROC evidence |
| Experiment root | Configuration, combined shadow predictions and audit/reload reports where generated |

Keep complete selected result folders for review. Root-level legacy models and figures are not the authoritative outputs for the cleaned notebooks. Archive older evidence outside the active working files rather than deleting the only copy.

## Recorded results

These are historical runs, not guaranteed outputs of a fresh execution.

| Linkage baseline (2026-09-23) | Test accuracy | Test F1 |
|---|---:|---:|
| Siamese fixed distance | 0.917500 | 0.914970 |
| Siamese linkage head | 0.935875 | 0.936027 |
| Direct-embedding MLP | 0.899875 | 0.902305 |

All three use the external 8,000-pair target test set. The direct MLP threshold was 0.55, selected on a training-derived validation subset. Siamese encoder/head validation uses 10% of the supplied training split. Direct MLP uses 72% for optimisation, 8% to monitor training and 20% for threshold selection. The baseline notebook does not fix all random seeds, so fresh results vary.

| Membership attack (2026-09-20) | Training examples | Target accuracy | ROC AUC |
|---|---:|---:|---:|
| K=5 | 67200 | 0.527753 | 0.542242 |
| K=10 | 134400 | 0.527902 | 0.540738 |

The attack exhibits weak discrimination in this experiment. Near-chance performance does not prove privacy or security. Increasing K also increases attack-training data; this is not an isolated test of K. Individual IDs overlap between pools/splits despite pair separation. Conclusions concern pair membership, not entity-level independence. Repeated seeds and uncertainty estimates remain future work.

## Troubleshooting

- Missing module: check the notebook interpreter, then install requirements into the selected fresh environment.
- TypeAliasType / typing_extensions errors: use the pinned IPython/kernel stack; avoid mixing it with modern Jupyter packages in the same environment.
- Missing saved run: restore the full experiment folder or intentionally select new training mode.
- CSV shape/label assertion failure: check the source and row alignment; do not suppress the assertion.
- TensorFlow retracing warnings: may reflect multiple model instances; inspect failures separately. A warning is not proof that saved-model verification failed.
- High memory use: use the supplied embeddings, close unnecessary apps, and avoid parallel training. Do not regenerate BERT embeddings during normal notebook execution.

## Verification status and sources

The final review checked Python syntax, notebook structure, saved outputs and evaluation code. The cleaned notebook copies retain executable behaviour and historical outputs. Full neural training and native Windows/Mac installation were not rerun in the review environment; run the installation checks above on each platform. See `FINAL_REVIEW.md` for precise edits and limitations.

Package sources: [TensorFlow 2.13.1](https://pypi.org/project/tensorflow/2.13.1/), [Apple Silicon TensorFlow 2.13.1](https://pypi.org/project/tensorflow-macos/2.13.1/), [Windows-compatible TensorFlow IO 0.31.0](https://pypi.org/project/tensorflow-io-gcs-filesystem/0.31.0/). Dependencies preserve this project's historical model stack; this is not a recommendation to use that stack for unrelated new projects.
