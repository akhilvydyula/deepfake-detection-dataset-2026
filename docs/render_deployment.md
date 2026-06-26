# Deploy The Streamlit App On Render

This project includes a Render blueprint in `render.yaml`.

## Render Settings

Use these settings if you create the service manually:

- Service type: Web Service
- Runtime: Python
- Build command: `make install`
- Start command: `make render-start`
- Python version: `python-3.11.9` from `runtime.txt`

## Required Model Checkpoint

The app needs a trained checkpoint to make predictions. By default it looks for:

```text
models/best_resnet18.pt
```

Model files are ignored by Git, so Render will not receive your local checkpoint automatically.

Recommended deployment options:

1. Upload the trained checkpoint to a Render persistent disk and set `MODEL_CHECKPOINT_PATH` to that path.
2. Store the checkpoint in private object storage, download it into the service using Render Shell, and set `MODEL_CHECKPOINT_PATH`.
3. If the checkpoint is small enough and safe to publish, explicitly add it to Git by changing `.gitignore`; this is usually not recommended for ML model files.

## Environment Variables

Set these in Render:

```text
APP_CONFIG_PATH=configs/default.yaml
MODEL_CHECKPOINT_PATH=/path/to/best_resnet18.pt
```

If `MODEL_CHECKPOINT_PATH` is blank, the app falls back to the checkpoint configured in `configs/default.yaml`.

## Deploy Steps

1. Push this repository to GitHub.
2. In Render, create a new Web Service or Blueprint from the repo.
3. Confirm the build command is `make install`.
4. Confirm the start command is `make render-start`.
5. Add `MODEL_CHECKPOINT_PATH` once your trained model is available on Render.
6. Deploy.

## Local Check Before Deploying

Run:

```powershell
make validate
make app
```

The local app should work before deploying. If the local app says the checkpoint is missing, Render will also need the checkpoint before predictions can run.
