# A small autoresearch workshop

Run a small byte-level language model on a laptop CPU, try one configuration change, then let an existing coding agent investigate up to three candidates.

## Run the notebook

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then run:

```sh
git clone https://github.com/saeejithnair/hack-the-valley-autoresearch.git
cd hack-the-valley-autoresearch
uv sync --python 3.12 --locked
uv run jupyter lab Live-Workshop.ipynb
```

For Windows setup, see [WINDOWS.md](WINDOWS.md). The dataset is bundled. Installation downloads dependencies. Python training makes no model API calls; your coding agent requires its own account.

Run the notebook in order. It measures a fresh baseline, guides a manual batch-size experiment, and restores that edit before the agent exercise. Open your agent in the same folder and give it the prompt in the notebook. `program.md` describes the bounded investigation.

The agent edits `train.py`. Keep `experiment.py`, `prepare.py`, the dataset, and dependencies unchanged. JSON measurements go under ignored `runs/`; the agent records hypotheses, evidence, decisions, and next steps in ignored `results.tsv`. Candidate code is preserved in local Git commits. Configure your Git name and email before agent experiments.

## What the score means

Each candidate trains new weights from scratch for ten seconds on two CPU threads. The fixed evaluator measures next-byte predictions on 4,096 held-out examples. Validation bits per byte, or BPB, is lower for better predictions. Uniform guessing over 256 byte values scores eight BPB.

These are short, time-limited comparisons. Fixed seeds do not remove timing variation, small score differences need repetition, and repeated selection can overfit validation data. A negative result is valid evidence. There is no guaranteed improvement.

This implementation and notebook were executed on Apple Silicon macOS. Windows and Linux dependencies are locked; execution on those platforms has not been verified here. Compare configurations within a session on your own machine.

Inspired by [Karpathy’s autoresearch](https://github.com/karpathy/autoresearch), with a separate small CPU implementation. Dataset attribution is in [THIRD_PARTY.md](THIRD_PARTY.md).
