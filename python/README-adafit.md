# AdaFit bridge

This folder contains a thin bridge to integrate AdaFit inference with `visibilityLattices`.

The bridge runs in strict mode: if AdaFit fails, it returns an error immediately.

## What is included

- `adafit_runner.py`: runs official AdaFit inference from a local checkout.
- `test_adafit_runner.py`: strict-mode smoke test (runner must fail when `--repo` is missing).
- `requirements-adafit.txt`: Python dependencies needed by official AdaFit mode.

## Expected workflow

1. Clone the official AdaFit repository: `https://github.com/Runsong123/AdaFit`.
2. Use a trained checkpoint (the official repo already contains one under `trained_model/AdaFit/`).
3. Point the UI or CLI options of `visibilityLattices` to:
   - the Python executable,
   - this runner script,
   - the local AdaFit repository,
   - and optionally the checkpoint file.

## Official AdaFit mode

Preferred mode is direct inference from a local checkout:

```bash
python3 python/adafit_runner.py \
  --input /tmp/pts.xyz \
  --output /tmp/normals.xyz \
  --repo /absolute/path/to/AdaFit
```

Optional arguments:

- `--checkpoint`: defaults to `trained_model/AdaFit/my_experiment_model_599.pth`
- `--params`: defaults to `*_params.pth` inferred from checkpoint name
- `--model-scale`: `auto|single|multi`
- `--batch-size`: inference batch size
- `--device`: `auto|cpu|cuda`

## Smoke tests

```bash
python3 python/test_adafit_runner.py
python3 python/adafit_runner.py --input /tmp/pts.xyz --output /tmp/normals.xyz --repo /absolute/path/to/AdaFit
```



