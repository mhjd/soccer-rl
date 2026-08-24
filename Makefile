PYTHON := .venv/bin/python
MJ_PYTHON := .venv/bin/mjpython
LOCOMOTION_PYTHON := .venv-locomotion/bin/python
LOCOMOTION_MJ_PYTHON := .venv-locomotion/bin/mjpython
LOCOMOTION_UV_CACHE ?= /tmp/soccer-rl-uv-cache
SEED ?= 0
EPISODE ?= 0
ALGORITHM ?= sac
TIMESTEPS ?= 200000
EPISODES ?= 100
REWARD_MODE ?= approach_progress
MODEL_LABEL ?= benchmark

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
	check-g1-locomotion inspect-g1-locomotion \
	check-g1-soccer inspect-g1-soccer \
	check-g1-soccer-env inspect-g1-soccer-env \
	train-g1-soccer-policy evaluate-g1-soccer-policy \
	train-g1-soccer-algorithm-benchmark \
	train-g1-soccer-sac-goal-only \
	evaluate-g1-soccer-algorithm-benchmark \
	evaluate-g1-soccer-sac-goal-only \
	evaluate-g1-soccer-geometric-benchmark \
	evaluate-g1-soccer-broad-algorithm-benchmark \
	evaluate-g1-soccer-sac-goal-only-broad \
	evaluate-g1-soccer-broad-geometric-benchmark \
	train-g1-soccer-curriculum evaluate-g1-soccer-randomized-policy \
	train-g1-soccer-recovery evaluate-g1-soccer-recovery \
	train-g1-soccer-foot-approach \
	finetune-g1-soccer-foot-approach \
	evaluate-g1-soccer-foot-approach \
	train-g1-soccer-executable-commands \
	evaluate-g1-soccer-executable-commands \
	inspect-g1-policy-vs-geometric \
	evaluate-g1-geometric-broad \
	inspect-g1-geometric-broad \
	check-g1-kick-residual \
	evaluate-g1-kick-challenge \
	evaluate-g1-kick-zero-residual \
	train-g1-kick-residual \
	evaluate-g1-kick-residual \
	evaluate-g1-geometric-kick-residual \
	check-g1-high-level-kick-residual \
	evaluate-g1-high-level-kick-zero-residual \
	train-g1-high-level-kick-residual \
	evaluate-g1-high-level-kick-residual \
	evaluate-g1-geometric-combined-residuals

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
	@echo "  make check-g1-soccer                             Check scripted foot-ball interaction"
	@echo "  make inspect-g1-soccer                           Render the scripted G1 soccer rollout"
	@echo "  make check-g1-soccer-env                         Check the Gymnasium G1 soccer environment"
	@echo "  make inspect-g1-soccer-env                       Render an environment rollout"
	@echo "  make train-g1-soccer-policy                      Train the high-level G1 soccer policy"
	@echo "  make evaluate-g1-soccer-policy                   Render the learned high-level policy"
	@echo "  make train-g1-soccer-algorithm-benchmark         Train PPO or SAC on the shared benchmark"
	@echo "  make train-g1-soccer-sac-goal-only               Train the sparse-reward SAC benchmark"
	@echo "  make evaluate-g1-soccer-algorithm-benchmark      Evaluate one benchmark model"
	@echo "  make evaluate-g1-soccer-sac-goal-only            Evaluate sparse-reward SAC"
	@echo "  make evaluate-g1-soccer-geometric-benchmark      Evaluate the geometric reference on the benchmark"
	@echo "  make evaluate-g1-soccer-broad-algorithm-benchmark Evaluate a learned policy on broad hard starts"
	@echo "  make evaluate-g1-soccer-sac-goal-only-broad      Evaluate sparse-reward SAC on broad starts"
	@echo "  make evaluate-g1-soccer-broad-geometric-benchmark Evaluate geometric control on broad hard starts"
	@echo "  make train-g1-soccer-curriculum                  Continue training with adaptive random starts"
	@echo "  make evaluate-g1-soccer-randomized-policy        Render randomized learned-policy episodes"
	@echo "  make train-g1-soccer-recovery                    Continue training with progressive recovery starts"
	@echo "  make evaluate-g1-soccer-recovery                 Render forced recovery-start episodes"
	@echo "  make train-g1-soccer-foot-approach                Train with foot-aware approach shaping"
	@echo "  make finetune-g1-soccer-foot-approach             Focus training on difficult recovery starts"
	@echo "  make evaluate-g1-soccer-foot-approach             Render the improved recovery policy"
	@echo "  make train-g1-soccer-executable-commands          Train with executable walking commands"
	@echo "  make evaluate-g1-soccer-executable-commands       Render the executable-command policy"
	@echo "  make inspect-g1-policy-vs-geometric               Compare ten failed PPO starts side by side"
	@echo "  make evaluate-g1-geometric-broad                  Measure broad geometric-controller generalization"
	@echo "  make inspect-g1-geometric-broad EPISODE=N          Render one reproducible broad test case"
	@echo "  make check-g1-kick-residual                       Check the low-level residual interface"
	@echo "  make evaluate-g1-kick-challenge                   Compare low-level residuals on hard boundary cases"
	@echo "  make evaluate-g1-kick-zero-residual               Measure the isolated contact baseline"
	@echo "  make train-g1-kick-residual                        Train the low-level kick residual"
	@echo "  make evaluate-g1-kick-residual                     Evaluate the learned kick residual"
	@echo "  make evaluate-g1-geometric-kick-residual           Test the residual on geometric approaches"
	@echo "  make check-g1-high-level-kick-residual              Check the high-level residual interface"
	@echo "  make evaluate-g1-high-level-kick-zero-residual      Measure its isolated contact baseline"
	@echo "  make train-g1-high-level-kick-residual               Train the high-level command residual"
	@echo "  make evaluate-g1-high-level-kick-residual            Evaluate the learned command residual"
	@echo "  make evaluate-g1-geometric-combined-residuals        Test both residual levels together"

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

check-g1-soccer:
	$(LOCOMOTION_PYTHON) -m scripts.evaluate_3d_g1_soccer

inspect-g1-soccer:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_soccer --render

check-g1-soccer-env:
	$(LOCOMOTION_PYTHON) -m scripts.check_3d_g1_soccer_env

inspect-g1-soccer-env:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.check_3d_g1_soccer_env --render

train-g1-soccer-policy:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--seed $(SEED)

evaluate-g1-soccer-policy:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--seed $(SEED) \
		--render

train-g1-soccer-algorithm-benchmark:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--algorithm $(ALGORITHM) \
		--timesteps $(TIMESTEPS) \
		--seed $(SEED) \
		--randomize-initial-positions \
		--observation-mode soccer_state \
		--reward-mode $(REWARD_MODE) \
		--max-episode-steps 200 \
		--checkpoint-frequency 20000 \
		--output models/$(ALGORITHM)_3d_g1_soccer_$(MODEL_LABEL)_seed$(SEED).zip

train-g1-soccer-sac-goal-only:
	$(MAKE) train-g1-soccer-algorithm-benchmark \
		ALGORITHM=sac REWARD_MODE=goal MODEL_LABEL=goal_only

evaluate-g1-soccer-algorithm-benchmark:
	$(LOCOMOTION_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--algorithm $(ALGORITHM) \
		--model models/$(ALGORITHM)_3d_g1_soccer_$(MODEL_LABEL)_seed$(SEED).zip \
		--episodes $(EPISODES) \
		--seed 100000 \
		--randomize-initial-positions \
		--observation-mode soccer_state \
		--max-episode-steps 200

evaluate-g1-soccer-sac-goal-only:
	$(MAKE) evaluate-g1-soccer-algorithm-benchmark \
		ALGORITHM=sac MODEL_LABEL=goal_only

evaluate-g1-soccer-geometric-benchmark:
	$(LOCOMOTION_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--algorithm geometric \
		--episodes $(EPISODES) \
		--seed 100000 \
		--randomize-initial-positions \
		--observation-mode soccer_state \
		--max-episode-steps 200

evaluate-g1-soccer-broad-algorithm-benchmark:
	$(LOCOMOTION_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--algorithm $(ALGORITHM) \
		--model models/$(ALGORITHM)_3d_g1_soccer_$(MODEL_LABEL)_seed$(SEED).zip \
		--episodes $(EPISODES) \
		--seed 100000 \
		--broad-initial-positions \
		--observation-mode soccer_state \
		--max-episode-steps 500

evaluate-g1-soccer-sac-goal-only-broad:
	$(MAKE) evaluate-g1-soccer-broad-algorithm-benchmark \
		ALGORITHM=sac MODEL_LABEL=goal_only

evaluate-g1-soccer-broad-geometric-benchmark:
	$(LOCOMOTION_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--algorithm geometric \
		--episodes $(EPISODES) \
		--seed 100000 \
		--broad-initial-positions \
		--observation-mode soccer_state \
		--max-episode-steps 500

train-g1-soccer-curriculum:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--seed $(SEED) \
		--adaptive-curriculum \
		--resume models/ppo_3d_g1_soccer_fixed.zip \
		--output models/ppo_3d_g1_soccer_randomized.zip

evaluate-g1-soccer-randomized-policy:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--model models/ppo_3d_g1_soccer_randomized.zip \
		--seed $(SEED) \
		--randomize-initial-positions \
		--episodes 5 \
		--render

train-g1-soccer-recovery:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--seed $(SEED) \
		--recovery-curriculum \
		--max-episode-steps 200 \
		--resume models/ppo_3d_g1_soccer_randomized.zip \
		--output models/ppo_3d_g1_soccer_recovery.zip

evaluate-g1-soccer-recovery:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--model models/ppo_3d_g1_soccer_recovery.zip \
		--seed $(SEED) \
		--randomize-initial-positions \
		--recovery-start-probability 1.0 \
		--max-episode-steps 200 \
		--episodes 5 \
		--render

train-g1-soccer-foot-approach:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--seed $(SEED) \
		--recovery-curriculum \
		--observation-mode soccer_state \
		--reward-mode approach_progress \
		--learning-rate 0.00003 \
		--target-kl 0.02 \
		--max-episode-steps 200 \
		--transfer-from models/ppo_3d_g1_soccer_recovery.zip \
		--output models/ppo_3d_g1_soccer_foot_approach.zip

finetune-g1-soccer-foot-approach:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--seed $(SEED) \
		--recovery-curriculum \
		--initial-curriculum-difficulty 0.8 \
		--observation-mode soccer_state \
		--reward-mode approach_progress \
		--max-episode-steps 200 \
		--resume models/ppo_3d_g1_soccer_foot_approach.zip \
		--output models/ppo_3d_g1_soccer_foot_approach_finetuned.zip

evaluate-g1-soccer-foot-approach:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--model models/ppo_3d_g1_soccer_foot_approach_finetuned.zip \
		--seed $(SEED) \
		--observation-mode soccer_state \
		--randomize-initial-positions \
		--recovery-start-probability 1.0 \
		--max-episode-steps 200 \
		--episodes 5 \
		--render

train-g1-soccer-executable-commands:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_soccer \
		--seed $(SEED) \
		--recovery-curriculum \
		--initial-curriculum-difficulty 0.8 \
		--observation-mode soccer_state \
		--reward-mode approach_progress \
		--max-episode-steps 200 \
		--resume models/ppo_3d_g1_soccer_foot_approach_finetuned.zip \
		--output models/ppo_3d_g1_soccer_executable_commands.zip

evaluate-g1-soccer-executable-commands:
	$(LOCOMOTION_MJ_PYTHON) -m scripts.evaluate_3d_g1_soccer_policy \
		--model models/ppo_3d_g1_soccer_executable_commands.zip \
		--seed $(SEED) \
		--observation-mode soccer_state \
		--randomize-initial-positions \
		--recovery-start-probability 1.0 \
		--max-episode-steps 200 \
		--episodes 5 \
		--render

inspect-g1-policy-vs-geometric:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_geometric_controller \
		--seed $(SEED) \
		--episodes 100 \
		--failed-episodes 10 \
		--render-failures \
		--playback-speed 2

evaluate-g1-geometric-broad:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_geometric_generalization \
		--seed $(SEED) \
		--episodes 100 \
		--report-json /tmp/soccer-rl-g1-geometric-seed$(SEED).json

inspect-g1-geometric-broad:
	$(LOCOMOTION_MJ_PYTHON) \
		-m scripts.inspect_3d_g1_geometric_generalization \
		--seed $(SEED) \
		--episode $(EPISODE)

check-g1-kick-residual:
	$(LOCOMOTION_PYTHON) -m scripts.check_3d_g1_residual_interface

evaluate-g1-kick-challenge:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_kick_challenge \
		--gait-timings 101 \
		--seed $(SEED)

evaluate-g1-kick-zero-residual:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_kick_residual \
		--seed $(SEED) \
		--episodes 1000

train-g1-kick-residual:
	$(LOCOMOTION_PYTHON) -m scripts.train_3d_g1_kick_residual

evaluate-g1-kick-residual:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_kick_residual \
		--model models/ppo_3d_g1_kick_residual.zip \
		--seed $(SEED) \
		--episodes 1000

evaluate-g1-geometric-kick-residual:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_geometric_generalization \
		--residual-model models/ppo_3d_g1_kick_residual_100k.zip \
		--seed $(SEED) \
		--episodes 100 \
		--report-json /tmp/soccer-rl-g1-residual-seed$(SEED).json

check-g1-high-level-kick-residual:
	$(LOCOMOTION_PYTHON) -m scripts.check_3d_g1_high_level_residual

evaluate-g1-high-level-kick-zero-residual:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_high_level_kick_residual \
		--seed $(SEED) \
		--episodes 1000

train-g1-high-level-kick-residual:
	$(LOCOMOTION_PYTHON) \
		-m scripts.train_3d_g1_high_level_kick_residual

evaluate-g1-high-level-kick-residual:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_high_level_kick_residual \
		--model models/ppo_3d_g1_high_level_kick_residual.zip \
		--seed $(SEED) \
		--episodes 1000

evaluate-g1-geometric-combined-residuals:
	$(LOCOMOTION_PYTHON) \
		-m scripts.evaluate_3d_g1_geometric_generalization \
		--residual-model models/ppo_3d_g1_kick_residual.zip \
		--high-level-residual-model models/ppo_3d_g1_high_level_kick_residual.zip \
		--seed $(SEED) \
		--episodes 100 \
		--report-json /tmp/soccer-rl-g1-combined-residuals-seed$(SEED).json
