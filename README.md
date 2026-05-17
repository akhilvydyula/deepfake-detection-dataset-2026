# Deepfake Detection Dataset 2026

A small, reproducible baseline pipeline for the Kaggle [Deepfake Detection Dataset 2026](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026/data).

The dataset contains 6,557 face-image records with binary `REAL` / `FAKE` labels, image URLs, metadata, and split information. This repo downloads the CSV, fetches the referenced images, generates basic EDA outputs, trains a ResNet-18 baseline, and evaluates it on the held-out split.

## Project Layout

```text
configs/default.yaml               Pipeline settings
scripts/download_kaggle_dataset.py Kaggle CSV download helper
src/deepfake_detection/            Reusable pipeline code
notebooks/                         EDA and baseline training notebooks
data/raw/                          Downloaded Kaggle files
data/processed/                    Downloaded images and enriched metadata
models/                            Trained checkpoints
reports/                           EDA and evaluation outputs
```

Large data, generated reports, and model files are ignored by Git.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Configure Kaggle API credentials before downloading. On Windows, place your `kaggle.json` token at:

```text
C:\Users\<your-user>\.kaggle\kaggle.json
```

## Run The Pipeline

Download and extract the Kaggle dataset:

```powershell
python scripts/download_kaggle_dataset.py
```

Generate summary JSON and distribution plots:

```powershell
python -m deepfake_detection.eda
```

Download the image URLs into split/label folders and write enriched metadata:

```powershell
python -m deepfake_detection.download_images
```

Train the baseline model:

```powershell
python -m deepfake_detection.train
```

Evaluate the best checkpoint:

```powershell
python -m deepfake_detection.evaluate --split test
```

## Notebooks

Launch Jupyter from the repo root:

```powershell
jupyter notebook
```

Available notebooks:

- `notebooks/01_dataset_eda.ipynb`: inspect metadata, label balance, split distributions, missing values, and EDA plots.
- `notebooks/02_baseline_training.ipynb`: run image download, ResNet-18 training, and test evaluation from notebook cells.

## Configuration

Edit `configs/default.yaml` to adjust paths, batch size, image size, epoch count, learning rate, worker count, or Kaggle dataset slug.

For quick smoke tests, set `download.limit` to a small number such as `100`. Leave it blank to process the full dataset.

## Baseline

The training script uses a ResNet-18 binary classifier with ImageNet pretrained weights by default. It expects `label_numeric` to use the dataset convention:

- `1`: `REAL`
- `0`: `FAKE`

If the CSV includes a valid split column, the pipeline uses it. If not, it creates a stratified 70% train, 10% validation, and 20% test split.

## Problems This Dataset Helps Solve

The most direct task is binary deepfake classification: train a model that takes a face image and predicts `REAL` (`1`) or `FAKE` (`0`). This is useful for social media moderation, fake-profile detection, media verification, and identity fraud prevention.

The dataset also supports production-style cybersecurity and KYC workflows. Synthetic faces can be used for fake accounts, scam profiles, bot identities, and fraudulent applications; a detector can provide a risk score before an account or identity check is approved.

Because the CSV includes metadata, this project can go beyond raw accuracy. Columns such as `gender`, `age_group`, `image_quality`, `detection_difficulty`, `confidence_score`, and `fake_method` make it possible to measure fairness, robustness, calibration, and generalization across generative methods.

## Research Directions

- Generalization across generators: split fake images by `fake_method` to test whether a detector overfits to StyleGAN3 artifacts or transfers to other GAN, diffusion, or future synthetic-face methods.
- Robustness to image quality and compression: compare high-quality and medium-quality subsets, then add augmentations for JPEG compression, resizing, blur, and lower-quality uploads.
- Difficulty-aware detection: use `detection_difficulty` to analyze easy, medium, and hard samples, benchmark human-level versus model performance, or experiment with curriculum learning and hard example mining.
- Fairness and demographic bias: compute per-group accuracy, precision, recall, false positive rate, and F1 across `gender` and `age_group`.
- Confidence calibration: use `confidence_score` and model probabilities to decide when predictions are reliable enough for high-risk workflows.
- Semi-supervised learning: use confidence scores as weak supervision or as filters for pseudo-labeling unlabeled real-world images.
- Out-of-distribution detection: treat unknown synthetic generators as open-set or anomaly-detection cases instead of only closed-set `REAL` / `FAKE` classification.
- Explainability and artifact localization: add Grad-CAM, saliency maps, attention heatmaps, or frequency-domain analysis to identify why an image was flagged.
- Transfer learning and domain adaptation: fine-tune pretrained vision or face-recognition models, then test transfer to datasets such as Celeb-DF or FaceForensics++.
- Real-world pipeline integration: use `image_url` to simulate an upload pipeline that downloads, preprocesses, scores, and flags images.

## Example Project Ideas

- Beginner: train a small CNN for binary `REAL` / `FAKE` classification.
- Intermediate: fine-tune ResNet, EfficientNet, or ConvNeXt with transfer learning.
- Advanced: compare CNNs against Vision Transformers and add explainability.
- Expert: deploy a FastAPI or Flask service that returns a deepfake probability for uploaded images.
- Research prompt: train a lightweight model, then analyze why it fails on hard samples, especially for the `50+` age group.

## Business Applications

- Fake account prevention for banking, dating, job, freelance, and social platforms.
- Misinformation detection for synthetic political or media campaigns.
- Trust and safety tooling for marketplaces and consumer platforms.
- Research benchmarking for students, ML engineers, and computer vision researchers.

## Key ML Challenges

- Moderate class imbalance: the dataset is about 57% fake and 43% real, so class weights, oversampling, focal loss, or threshold tuning may be useful.
- Overfitting to generator-specific artifacts.
- Dataset and subgroup bias.
- Robustness to compression and lower-quality uploads.
- Real-time inference speed.
- Explainability and confidence calibration.

## Outputs

- `reports/dataset_summary.json`: row counts, columns, and metadata distributions
- `reports/*_distribution.png`: basic EDA plots
- `data/processed/metadata_with_paths.csv`: source metadata plus local image paths and download status
- `models/best_resnet18.pt`: best validation F1 checkpoint
- `models/training_history.json`: epoch-level train and validation metrics
- `reports/test_metrics.json`: final evaluation metrics and confusion matrix

## License

The Kaggle dataset is listed as `CC0: Public Domain`. Check the source dataset page for the latest license and usage notes.
