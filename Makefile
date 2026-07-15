PYTHON := .venv/bin/python

.PHONY: help install train evaluate

help:
	@echo "Available targets:"
	@echo "  make install   Install project dependencies"
	@echo "  make train     Train the PPO policy"
	@echo "  make evaluate  Render one trained-policy episode"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m scripts.train

evaluate:
	$(PYTHON) -m scripts.evaluate
