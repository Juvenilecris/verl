set -x
ENGINE=${1:-vllm}

export SWANLAB_API_KEY="5fi2cgzDNC4ccLyceA7ni"           # 设置在线跟踪模式API
export SWANLAB_LOG_DIR="/data/wangnn/repos/verl/debiasing/log"  # 设置本地日志存储路径
# export SWANLAB_MODE="disabled"    # 包含四种模式：cloud云端跟踪模式（默认）、cloud-only仅云端跟踪本地不保存文件、local本地跟踪模式、disabled完全不记录用于debug

export CUDA_VISIBLE_DEVICES=5,6
data="data_bg"
train_path=/data/wangnn/repos/verl/debiasing/$data/train.parquet
val_path=/data/wangnn/repos/verl/debiasing/$data/validation.parquet


train_files="['$train_path', '$val_path']"

PYTHONUNBUFFRED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files=/data/wangnn/repos/verl/debiasing/$data/test.parquet \
    data.shuffle=True\
    data.train_batch_size=64\
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_multi_modal_inputs=False\
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.model.path=/data/wangnn/models/Qwen2.5-VL-3B-Instruct-backgrounding \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.weight_by_group=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.nccl_timeout=36000 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True\
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.calculate_tokens_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128\
    custom_reward_function.path=/data/wangnn/repos/verl/debiasing/waterbird_reward_func.py \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='debiasing_bird_bg' \
    trainer.experiment_name='qwen2_5_vl_3b_debiasing_v11' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5001 \
    trainer.total_epochs=10\
    trainer.val_before_train=False\
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2\
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2\
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=/data/wangnn/repos/verl/debiasing/waterbird_interaction.yaml $@
# actor_rollout_ref.actor.grad_clip=1.0 \
# kl约束
# entropy_coenf

    