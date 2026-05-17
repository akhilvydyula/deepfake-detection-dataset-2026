.PHONY: help install download eda images train evaluate pipeline notebook validate clean

PYTHON ?= python
CONFIG ?= configs/default.yaml
SPLIT ?= test

help:
	@echo "Deepfake Detection Dataset 2026"
	@echo ""
	@echo "Targets:"
	@echo "  make install    Install requirements and editable package"
	@echo "  make download   Download and extract Kaggle dataset CSV"
	@echo "  make eda        Generate dataset summary and EDA plots"
	@echo "  make images     Download image URLs into data/processed/images"
	@echo "  make train      Train the ResNet-18 baseline"
	@echo "  make evaluate   Evaluate the best checkpoint on SPLIT=$(SPLIT)"
	@echo "  make pipeline   Run download, eda, images, train, and evaluate"
	@echo "  make notebook   Launch Jupyter Notebook"
	@echo "  make validate   Compile Python files and validate notebooks"
	@echo "  make clean      Remove generated data, reports, models, and caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

download:
	$(PYTHON) scripts/download_kaggle_dataset.py --config $(CONFIG)

eda:
	$(PYTHON) -m deepfake_detection.eda --config $(CONFIG)

images:
	$(PYTHON) -m deepfake_detection.download_images --config $(CONFIG)

train:
	$(PYTHON) -m deepfake_detection.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m deepfake_detection.evaluate --config $(CONFIG) --split $(SPLIT)

pipeline: download eda images train evaluate

notebook:
	$(PYTHON) -m notebook

validate:
	$(PYTHON) -m compileall src scripts
	$(PYTHON) -c "import json; from pathlib import Path; [json.loads(path.read_text(encoding='utf-8')) for path in Path('notebooks').glob('*.ipynb')]; print('notebooks valid')"

clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for path in [Path('data/raw'), Path('data/interim'), Path('data/processed'), Path('models'), Path('reports')]]; [path.mkdir(parents=True, exist_ok=True) for path in [Path('data/raw'), Path('data/interim'), Path('data/processed'), Path('models'), Path('reports')]]"
