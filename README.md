# Multi-Turn Censorship Game

Two-player multi-turn censorship game trained via adversarial self-play (equilibrium training).

## Setup (uv)

```bash
cd /home/zhuochen/Projects/ee599_project
uv venv
source .venv/bin/activate
uv sync
```

## Config overview (`configs/default.yaml`)

- **model.defender / model.adversary**
  - `model_name`: HuggingFace model id (default: `gpt2`)
  - `max_length`: max prompt length (tokens)
  - `temperature`: sampling temperature
  - `use_learned_policy`: must be `true` for learned neural policies

- **game**
  - `max_turns`: turns per episode
  - `safety_threshold`: environment threshold (kept for compatibility)
  - `refuse_threshold`: defender decision threshold (used when not exploring)
  - `benign_prob`: probability an episode is benign during training
  - `lambda_violation`: penalty weight for unsafe outputs
  - `mu_refusal`: refusal penalty (benign + malicious)
  - `gamma_turn`: per-turn adversary cost
  - `terminate_on_violation`: end episode on first violation

- **training**
  - `num_episodes`, `batch_size`, `defender_lr`, `adversary_lr`, `gamma`
  - `eval_interval`, `eval_num_malicious`, `eval_num_benign`
  - `update_mode`: e.g. `alternating`
  - `warmstart_steps`: behavior cloning steps before RL
  - `checkpoint_dir`: output directory for checkpoints

- **data**
  - `train_datasets`: training dataset names (default includes `xguard`)
  - `val_datasets`: kept for compatibility (current `train.py` evaluates on XGuard-val)
  - `data_dir`, `split_seed`

- **logging**
  - `log_dir`, `log_interval`, `use_wandb`

## Train

```bash
source .venv/bin/activate
python train.py --config configs/default.yaml
```

Checkpoints are written under `checkpoints/<RUN_NAME>/` and logs under `logs/<RUN_NAME>/`.

## Evaluate (held-out MHJ)

```bash
source .venv/bin/activate
python evaluate.py \
  --config configs/default.yaml \
  --model_path checkpoints/<RUN_NAME> \
  --dataset mhj \
  --num_samples 500
```

