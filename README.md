# Flora

A training and evaluation pipeline for plant species classification in plain
PyTorch. MobileNetV3-Large with a replaced head, plus Optuna for architecture
search and a run-tracking layer that keeps every experiment comparable.

![Experiment comparison](docs/figs/experiments_comparison_metrics.png)

## What it does

- Trains MobileNetV3-Large with a modified dropout and classification head, at a
  configurable input size.
- Switches loss between cross-entropy and focal loss, and builds optimisers and
  schedulers per layer group so the head can train faster than the backbone.
- Runs a sanity check on sliced inputs before a long run, verifying tensor
  bounds, gradients, and parameter updates.
- Searches architecture and training hyperparameters with Optuna, writing each
  study into the run directory.
- Plots cross-entropy, top-1 accuracy, and macro F1 side by side across
  experiments, reading `training_history.json` from each run.

## Run it

```bash
git clone https://github.com/abderrahmenex86/Flora
cd Flora
pip install -r requirements.txt

python tools.py --mode verify
python main.py --mode test
python main.py --mode optimize
python main.py --mode train --learning_rate 0.001 --weight_decay 0.01 \
  --loss_type focal --num_workers_train 8
python main.py --mode infer \
  --run_directory artifacts/YYYYMMDD_HHMMSS_mobilenetv3_large \
  --image_path path/to/test_image.jpg
```

Resume an interrupted run with `python main.py --mode train --resume
--resume_directory artifacts/<run>`, and refresh the comparison dashboard with
`python tools.py --mode plot --run_directory artifacts/<run>`.

## How runs are tracked

Each run writes a directory under `artifacts/` containing the resolved
configuration, `training_history.json`, and checkpoints. The plotting tool walks
those directories, parses each `train.py` call with `ast` to recover the
arguments that produced the run, and builds the legend from them, so the
dashboard cannot drift from what was actually executed.

## Results

The repository publishes comparison plots rather than a results table. Per-run
metrics are in each `artifacts/<run>/training_history.json`, and the dashboard is
generated from those files.

## Limits

- One dataset. Species outside the training distribution fall back to the most
  similar class with no rejection path.
- The comparison dashboard is only as reliable as the run directories it reads;
  runs started outside `main.py` will not appear.
- No published accuracy figure, so results have to be reproduced from a run.

## Stack

Python · PyTorch · torchvision · Optuna · TorchMetrics · Matplotlib

## Related

- [Entomi](https://github.com/abderrahmenex86/Entomi) for insect and pest
  classification
- [Lesioni](https://github.com/abderrahmenex86/Lesioni) for lesion
  segmentation
