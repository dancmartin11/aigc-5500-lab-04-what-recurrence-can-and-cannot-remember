# AIGC-5500 Lab 4 - What Recurrence Can and Cannot Remember

**Student:** Daniel Alejandro Castillo Martin  
**Student ID:** N10019284  
**Course:** Advanced Deep Learning (AIGC-5500-0NA)  
**Professor:** Hossein Pourmodheji

---

Submission for Lab 04 of the Advanced Deep Learning course in the Artificial Intelligence with Machine Learning postgraduate program at Humber Polytechnic.

This project compares RNNs, LSTMs, and GRUs on a delayed-recall task, investigating gradients through time, gate initialization, gradient clipping, and the limits of recurrence.

## Setup

Python version: 3.11.15.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run

Open `daniel_alejandro_castillo_martin_LAB4.ipynb`, select the Python environment, and run the cells from top to bottom from the repository root.

The professor's starter is preserved unchanged in `scripts/lab04_starter.py`.

Figures and results are saved under `output/`.

## Task

The first sequence step contains a class label. Subsequent steps contain noise, and the model must recall the label at the final step.

There are four classes, so chance accuracy is 0.25.

## Starter Verification

The unchanged starter achieved 1.000 validation accuracy on the 10-step task after 600 updates and printed `Starter OK.`

## Experiments

The following sections will be added progressively:

1. Recurrent benchmark.
2. Gradients through time.
3. Forget-bias sweep and clipping comparison.
4. Timing and conclusions.