PYTHON := .venv/bin/python
MJ_PYTHON := .venv/bin/mjpython
SEED ?= 0

.PHONY: help install train evaluate \
	train-cylinder evaluate-cylinder \
	train-cylinder-combined evaluate-cylinder-combined \
	train-cylinder-phased evaluate-cylinder-phased \
	train-cylinder-combined-randomized \
	evaluate-cylinder-combined-randomized \
	train-cylinder-phased-randomized \
	evaluate-cylinder-phased-randomized \
	train-cylinder-approach-warmup-randomized \
	evaluate-cylinder-approach-warmup-randomized

help:
	@echo "Available targets:"
	@echo "  make install     Install project dependencies"
	@echo "  make train       Train the 2D PPO policy"
	@echo "  make evaluate    Render one trained 2D-policy episode"
	@echo "  make train-cylinder-combined    Train with both shaping terms"
	@echo "  make evaluate-cylinder-combined Render the combined policy"
	@echo "  make train-cylinder-phased      Train with contact-phased shaping"
	@echo "  make evaluate-cylinder-phased   Render the contact-phased policy"
	@echo "  make train-cylinder-combined-randomized    Train combined on random starts"
	@echo "  make evaluate-cylinder-combined-randomized Render combined on a random start"
	@echo "  make train-cylinder-phased-randomized      Train phased on random starts"
	@echo "  make evaluate-cylinder-phased-randomized   Render phased on a random start"
	@echo "  make train-cylinder-approach-warmup-randomized    Train light approach warmup"
	@echo "  make evaluate-cylinder-approach-warmup-randomized Render its learned policy"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m scripts.train_2d

evaluate:
	$(PYTHON) -m scripts.evaluate_2d

train-cylinder: train-cylinder-combined

evaluate-cylinder: evaluate-cylinder-combined

train-cylinder-combined:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--reward-strategy combined \
		--output models/ppo_3d_cylinder_combined.zip

evaluate-cylinder-combined:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_combined.zip \
		--reward-strategy combined \
		--episodes 1 \
		--render

train-cylinder-phased:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--reward-strategy contact_phased \
		--output models/ppo_3d_cylinder_contact_phased.zip

evaluate-cylinder-phased:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_contact_phased.zip \
		--reward-strategy contact_phased \
		--episodes 1 \
		--render

train-cylinder-combined-randomized:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--seed $(SEED) \
		--reward-strategy combined \
		--randomize-initial-positions \
		--output models/ppo_3d_cylinder_combined_randomized.zip

evaluate-cylinder-combined-randomized:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_combined_randomized.zip \
		--seed $(SEED) \
		--reward-strategy combined \
		--randomize-initial-positions \
		--episodes 1 \
		--render

train-cylinder-phased-randomized:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--seed $(SEED) \
		--reward-strategy contact_phased \
		--randomize-initial-positions \
		--output models/ppo_3d_cylinder_contact_phased_randomized.zip

evaluate-cylinder-phased-randomized:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_contact_phased_randomized.zip \
		--seed $(SEED) \
		--reward-strategy contact_phased \
		--randomize-initial-positions \
		--episodes 1 \
		--render

train-cylinder-approach-warmup-randomized:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--seed $(SEED) \
		--reward-strategy approach_warmup \
		--randomize-initial-positions \
		--output models/ppo_3d_cylinder_approach_warmup_80k_0p1_randomized.zip

evaluate-cylinder-approach-warmup-randomized:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_approach_warmup_80k_0p1_randomized.zip \
		--seed $(SEED) \
		--reward-strategy ball_goal_only \
		--randomize-initial-positions \
		--episodes 1 \
		--render
