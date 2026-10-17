# Windows: run the same experiment in PowerShell

The experiment runs on your CPU. You do not need an NVIDIA GPU, CUDA, WSL, or Docker for the Python training/notebook portion. Your coding agent needs its own working account and Windows environment; configure it using its normal instructions before the workshop.

This starter has been executed on Apple Silicon macOS. Dependency resolution against the locked versions passed for both Windows x64 and ARM64 with Python 3.12, including CPU PyTorch and the notebook dependencies. A Windows execution has not been performed here. Scores and throughput will differ across machines. Compare configurations on your own machine.

## 1. Install the prerequisites if needed

Open PowerShell. If `uv --version` or `git --version` is missing, install that tool:

```powershell
winget install --id=astral-sh.uv -e
winget install --id Git.Git -e --source winget
```

Close and reopen PowerShell after installation. If WinGet is unavailable, use the official [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) and [Git for Windows installer](https://git-scm.com/install/windows).

## 2. Clone the workshop

```powershell
git clone https://github.com/saeejithnair/hack-the-valley-autoresearch.git
cd hack-the-valley-autoresearch
```

The clone already contains an initial Git commit. Before the agent exercise, configure your Git name and email if needed:

```powershell
git config user.name "Your Name"
git config user.email "you@example.com"
```

## 3. Install and run

```powershell
uv sync --python 3.12 --locked
uv run jupyter lab Live-Workshop.ipynb
```

`uv` can download Python 3.12 when required. The first sync downloads dependencies, including PyTorch; the small starter ZIP does not contain those dependencies. The project explicitly selects the CPU PyTorch index on Windows and Linux.

Use `uv run` throughout. You do not need to activate `.venv`, run `Activate.ps1`, or change PowerShell execution policy. The notebook launches the same Python experiment through `subprocess`; it does not depend on Bash commands.

The evaluator refuses to overwrite an existing result. For a repetition, choose a fresh path, such as `runs/baseline-repeat-1.json`. The notebook automatically creates timestamped output names.

## 4. Run the agent

Open your already configured coding agent in this same folder. It must be able to edit `train.py`, run `uv run experiment.py`, read the output, and create local Git commits. Give it the prompt in the README or notebook. The Python training itself makes no model API calls; the coding agent uses its own model access.

If your agent runs inside WSL, keep the entire project, Python environment, and commands inside that environment. Do not alternate Windows and WSL interpreters against one shared virtual environment.

For the thirty-minute workshop, follow the live run if installation or agent authentication is not already working.

Sources: [uv installation](https://docs.astral.sh/uv/getting-started/installation/), [Git for Windows](https://git-scm.com/install/windows), [uv with CPU PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
