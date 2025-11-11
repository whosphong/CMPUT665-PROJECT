#!/bin/bash -ex

# --- Parameters to loop over ---
ENVS=("HalfCheetah-v5" "Ant-v5")
SEEDS=(26)
ALGOS=("ppo" "trpo")

# PPO hyperparameter grid
PPO_SAMPLE_SIZES=(2048)
PPO_CLIP_PARAMS=(0.2 0.3)
PPO_POLICY_LRS=(3e-4)
PPO_TARGET_KLS=(0.01 0.02)
PPO_TRAIN_POLICY_ITERS=(10 20)
PPO_TRAIN_VF_ITERS=(10 20)

# TRPO hyperparameter grid
TRPO_SAMPLE_SIZES=(2048)
TRPO_DELTAS=(0.01 0.02)
TRPO_BACKTRACK_COEFFS=(1.0 0.8)
TRPO_TRAIN_VF_ITERS=(10 80 120)

# Shared experiment defaults for hyperparameter tuning
COMMON_ARGS=(
  "--phase=train"
  "--total_train_steps=1000000"
  "--steps_per_iter=5000"
  "--evaluation_mode=tune"
  "--tune_eval_window_episodes=100"
  "--eval_freq=50000"
  "--checkpoint_freq=200000"
  "--tensorboard"
)

# --- SLURM script to call ---
SBATCH_SCRIPT="sbatch.sh"

sanitize() {
  echo "$1" | tr '.' 'p'
}

echo "Launching SLURM hyperparameter grid..."

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      case "$algo" in
        ppo)
          for sample_size in "${PPO_SAMPLE_SIZES[@]}"; do
            for clip_param in "${PPO_CLIP_PARAMS[@]}"; do
              for policy_lr in "${PPO_POLICY_LRS[@]}"; do
                for target_kl in "${PPO_TARGET_KLS[@]}"; do
                  for train_policy_iters in "${PPO_TRAIN_POLICY_ITERS[@]}"; do
                    for train_vf_iters in "${PPO_TRAIN_VF_ITERS[@]}"; do
                      job_suffix="ss$(sanitize "$sample_size")-cp$(sanitize "$clip_param")-plr$(sanitize "$policy_lr")-tkl$(sanitize "$target_kl")-tpi$(sanitize "$train_policy_iters")-tvi$(sanitize "$train_vf_iters")"
                      job_name="ppo-${env}-s${seed}-${job_suffix}"

                      extra_args=(
                        "${COMMON_ARGS[@]}"
                        "--ppo_sample_size=${sample_size}"
                        "--ppo_clip_param=${clip_param}"
                        "--ppo_policy_lr=${policy_lr}"
                        "--ppo_vf_lr=${policy_lr}"
                        "--ppo_target_kl=${target_kl}"
                        "--ppo_train_policy_iters=${train_policy_iters}"
                        "--ppo_train_vf_iters=${train_vf_iters}"
                      )

                      extra_arg_string="${extra_args[*]}"

                      echo "Submitting job: ${job_name}"
                      sbatch \
                        --job-name="${job_name}" \
                        "--export=ALL,ENV_NAME=${env},ALGO_NAME=ppo,SEED=${seed},EXTRA_ARGS=${extra_arg_string}" \
                        "${SBATCH_SCRIPT}"
                    done
                  done
                done
              done
            done
          done
          ;;
        trpo)
          for sample_size in "${TRPO_SAMPLE_SIZES[@]}"; do
            for delta in "${TRPO_DELTAS[@]}"; do
              for backtrack_coeff in "${TRPO_BACKTRACK_COEFFS[@]}"; do
                for train_vf_iters in "${TRPO_TRAIN_VF_ITERS[@]}"; do
                  job_suffix="ss$(sanitize "$sample_size")-dl$(sanitize "$delta")-bt$(sanitize "$backtrack_coeff")-tvi$(sanitize "$train_vf_iters")"
                  job_name="trpo-${env}-s${seed}-${job_suffix}"

                  extra_args=(
                    "${COMMON_ARGS[@]}"
                    "--trpo_sample_size=${sample_size}"
                    "--trpo_delta=${delta}"
                    "--trpo_backtrack_coeff=${backtrack_coeff}"
                    "--trpo_backtrack_iter=10"
                    "--trpo_backtrack_alpha=0.5"
                    "--trpo_train_vf_iters=${train_vf_iters}"
                  )

                  extra_arg_string="${extra_args[*]}"

                  echo "Submitting job: ${job_name}"
                  sbatch \
                    --job-name="${job_name}" \
                    "--export=ALL,ENV_NAME=${env},ALGO_NAME=trpo,SEED=${seed},EXTRA_ARGS=${extra_arg_string}" \
                    "${SBATCH_SCRIPT}"
                done
              done
            done
          done
          ;;
        *)
          echo "Skipping unsupported algo: ${algo}"
          ;;
      esac
    done
  done
done

echo "All jobs submitted."