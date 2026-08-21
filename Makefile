PYTHON := .venv/bin/python
MJ_PYTHON := .venv/bin/mjpython
LOCOMOTION_PYTHON := .venv-locomotion/bin/python
LOCOMOTION_MJ_PYTHON := .venv-locomotion/bin/mjpython
LOCOMOTION_UV_CACHE ?= /tmp/soccer-rl-uv-cache
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
	evaluate-cylinder-approach-warmup-randomized \
	train-cylinder-adaptive-curriculum \
	evaluate-cylinder-adaptive-curriculum \
	train-cylinder-curriculum-phased \
	evaluate-cylinder-curriculum-phased \
	measure-cylinder-adaptive-curve \
	measure-cylinder-phased-curve \
	measure-cylinder-curriculum-phased-curve \
	plot-cylinder-learning-curves \
	check-g1 inspect-g1 \
	setup-g1-locomotion download-g1-locomotion-policy \
	check-g1-locomotion inspect-g1-locomotion

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
	@echo "  make train-cylinder-adaptive-curriculum          Train sparse reward with adaptive starts"
	@echo "  make evaluate-cylinder-adaptive-curriculum       Render it on the full start distribution"
	@echo "  make train-cylinder-curriculum-phased            Train adaptive starts with phased shaping"
	@echo "  make evaluate-cylinder-curriculum-phased         Render the combined policy"
	@echo "  make measure-cylinder-adaptive-curve             Measure adaptive learning for one seed"
	@echo "  make measure-cylinder-phased-curve               Measure phased learning for one seed"
	@echo "  make measure-cylinder-curriculum-phased-curve    Measure curriculum plus phased shaping"
	@echo "  make plot-cylinder-learning-curves               Plot all measured seeds"
	@echo "  make check-g1                                    Run the headless G1 smoke simulation"
	@echo "  make inspect-g1                                  Render the isolated G1 model"
	@echo "  make setup-g1-locomotion                         Install isolated locomotion dependencies"
	@echo "  make download-g1-locomotion-policy               Download and verify the official policy"
	@echo "  make check-g1-locomotion                         Check stand, forward, lateral, and yaw commands"
	@echo "  make inspect-g1-locomotion                       Render one forward-command rollout"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m scripts.train_2d

evaluate:
	$(PYTHON) -m scripts.evaluate_2d

train-cylinder: train-cylinder-adaptive-curriculum

evaluate-cylinder: evaluate-cylinder-adaptive-curriculum

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

train-cylinder-adaptive-curriculum:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--seed $(SEED) \
		--reward-strategy goal_only \
		--adaptive-curriculum \
		--output models/ppo_3d_cylinder_adaptive_curriculum.zip

evaluate-cylinder-adaptive-curriculum:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_adaptive_curriculum.zip \
		--seed $(SEED) \
		--reward-strategy goal_only \
		--randomize-initial-positions \
		--episodes 1 \
		--render

train-cylinder-curriculum-phased:
	$(PYTHON) -m scripts.train_3d_cylinder \
		--seed $(SEED) \
		--reward-strategy contact_phased \
		--adaptive-curriculum \
		--output models/ppo_3d_cylinder_curriculum_contact_phased.zip

evaluate-cylinder-curriculum-phased:
	$(MJ_PYTHON) -m scripts.evaluate_3d_cylinder \
		--model models/ppo_3d_cylinder_curriculum_contact_phased.zip \
		--seed $(SEED) \
		--reward-strategy goal_only \
		--randomize-initial-positions \
		--episodes 1 \
		--render

measure-cylinder-adaptive-curve:
	$(PYTHON) -m scripts.measure_3d_learning_curve \
		--strategy adaptive_curriculum \
		--seed $(SEED)

measure-cylinder-phased-curve:
	$(PYTHON) -m scripts.measure_3d_learning_curve \
		--strategy contact_phased \
		--seed $(SEED)

measure-cylinder-curriculum-phased-curve:
	$(PYTHON) -m scripts.measure_3d_learning_curve \
		--strategy curriculum_contact_phased \
		--seed $(SEED)

plot-cylinder-learning-curves:
	$(PYTHON) -m scripts.plot_3d_learning_curves

check-g1:
	$(PYTHON) -m scripts.inspect_3d_g1

inspect-g1:
	$(MJ_PYTHON) -m scripts.inspect_3d_g1 --duration 10 --render

setup-g1-locomotion:
	UV_CACHE_DIR=$(LOCOMOTION_UV_CACHE) uv venv .venv-locomotion --python 3.11
	UV_CACHE_DIR=$(LOCOMOTION_UV_CACHE) uv pip install \
		--python $(LOCOMOTION_PYTHON) \
		-r requirements-locomotion.txt

download-g1-locomotion-policy:
	$(PYTHON) -m scripts.download_g1_locomotion_policy

check-g1-locomotion:
	$(LOCOMOTION_PYTHON) -m scripts.evaluate_3d_g1_locomotion --suite

inspect-g1-locomotion:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_locomotion \
		--vx 0.5 \
		--duration 10 \
		--render
