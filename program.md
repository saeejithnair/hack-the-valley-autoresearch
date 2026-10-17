# Bounded autoresearch session

Investigate whether changes to a small byte-level language model can reduce validation bits per byte within the fixed ten-second CPU training budget. Lower `val_bpb` is better. Complete **at most three candidate attempts**, then stop.

## Research boundary

- Edit **only `train.py`**: the model factory, optimizer, and `BATCH_SIZE` are the available research surface.
- Preserve the interface: `build_model(vocab_size, context_length)` returns a CPU model producing `[batch, 256]` next-byte logits; `build_optimizer(model)` returns its optimizer. Keep the model small enough for this short CPU exercise.
- Do not change data, splits, context length, CPU thread count, training budget, evaluation, dependencies, or other repository files.
- You may write experiment artifacts under ignored `runs/` and the ignored research record `results.tsv`.
- Do not install new dependencies, fetch external data, access a GPU, or alter the evaluator to improve a reported score.
- Preserve candidate code in local Git commits. Do not use `git reset --hard`, delete history, or discard unrelated user work.

## Establish the starting evidence

1. Read `README.md`, `train.py`, `prepare.py`, and `experiment.py`.
2. Inspect Git status. Before starting, `train.py` must match the committed starting or best-retained version:

   ```sh
   git diff --exit-code HEAD -- train.py
   ```

   If it has uncommitted edits, leave them intact and explain that the experiment needs an identified starting training version. Other user edits, including notebook outputs saved during the walkthrough, are allowed. Preserve them and never include them in your commits.
3. Record the starting commit as the reference for the protected experiment inputs. Check these paths before starting and throughout the session:

   ```sh
   git diff --exit-code <starting-commit> -- experiment.py prepare.py data/input.txt pyproject.toml uv.lock
   ```

4. Inspect existing results and artifacts. If there is no baseline, run:

   ```sh
   uv run experiment.py --output runs/baseline.json
   ```

   Read the actual JSON output. Associate the result with its code SHA and Git commit. Do not reuse an artifact as evidence for different training code.
5. If previous experiments exist, preserve them and identify the best valid retained result and its code. Use unique experiment numbers so nothing is overwritten. Otherwise, the baseline is the initial best result.

## Record the investigation

Create `results.tsv` if absent, using this tab-separated header:

```text
commit	val_bpb	status	hypothesis	evidence	next_step
```

Record the baseline once with status `baseline`. Candidate statuses are `keep`, `discard`, or `crash`. Use one line per experiment and avoid tabs or line breaks inside fields. Include the artifact path in `evidence`.

For a crash, leave the score field empty. **Never report a crash as a score of zero.** Describe the actual error and what it implies for the next attempt.

## For each candidate

1. Start from the best retained training code. State one testable hypothesis and the change that will investigate it. Choose your own experiment; do not assume any change is guaranteed to improve the result.
2. Edit only `train.py`. Preserve the candidate before running it:

   ```sh
   git commit --only -m "Experiment: describe the tested change" -- train.py
   ```

3. Check fixed files against the starting commit, then run the unchanged evaluator with a new output path:

   ```sh
   uv run experiment.py --output runs/experiment-N.json
   ```

4. Read the actual result. Check the code SHA and inspect the score, training time, steps, and parameter count. If execution fails, preserve the error under `runs/` and record a crash. A crash counts toward the three-attempt limit.
   Bound each evaluator command to 60 seconds of wall time, including startup and evaluation. If it times out, terminate the experiment process before continuing and record the timeout as a crash. The ten-second training loop alone cannot interrupt a single pathological model operation.
5. Compare the candidate's `val_bpb` with the best valid retained result. Keep a strictly lower score provisionally. Discard an equal or worse score. Describe the size of the difference without treating it as proof of a general improvement.
6. Append the candidate commit, score, decision, hypothesis, measured evidence, and an evidence-informed next step to `results.tsv`. Distinguish what the experiment showed from your interpretation.
7. For a discarded or crashed candidate, restore only `train.py` from the best retained commit:

   ```sh
   git restore --source=<best-commit> -- train.py
   git commit --only -m "Restore best training candidate" -- train.py
   ```

   The candidate commit and its result remain available. Continue from the restored code. The path-specific commit preserves unrelated staged and unstaged user changes. Do not reset Git history.
8. Check fixed files again. If they changed, stop and explain; do not silently repair or hide the violation.

## Stop and explain

Stop after three candidate attempts, or sooner if the environment prevents a meaningful experiment. Leave `train.py` at the best retained version.

Report the baseline and best measured scores, the attempted hypotheses, the relevant artifact paths, and one proposed next investigation grounded in the results. Explain any crashes, uncertainty, or confounded comparisons. State what a person would need to inspect to assess your conclusion. Do not claim a breakthrough or guaranteed generalization from this short search.

Ask for human knowledge when you can explain what information would change the investigation. After stopping, wait for a new instruction before executing more experiments.
