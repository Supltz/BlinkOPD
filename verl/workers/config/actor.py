# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    pass

    full_logit_distillation: bool = True
    alpha: float = 0.0
    gamma: float = 1.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    teacher_update_interval: Optional[int] = None
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_always_on: bool = False
    teacher_model_source: str = "legacy"
    teacher_model_path: Optional[str] = None
    teacher_image_key: Optional[str] = None
    counterfactual_null_mode: Optional[str] = None
    counterfactual_extrapolation_beta: float = 1.0
    counterfactual_beta_gate: str = "none"
    counterfactual_gate_first_key: str = "vg_first_letter"
    counterfactual_gate_options_key: str = "vg_options"
    counterfactual_gate_on_scale: float = 1.0
    counterfactual_gate_off_scale: float = 0.0
    isolate_packed_sequences: bool = False
    skip_first_tokens: int = 0
    teacher_prompt_mode: Optional[str] = None
    answer_hint_template: str = (
        "\n\nHere is a reference solution to this problem:\n"
        "{answer}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
    )
    fallback_to_policy_loss_on_missing_teacher: bool = False
    log_prob_dump_dir: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        if self.gamma < 0.0:
            raise ValueError(f"self_distillation.gamma must be non-negative, got {self.gamma}")
        valid_teacher_regularization = ["frozen", "ema", "trust-region", "progressive"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.teacher_update_interval is not None and self.teacher_update_interval <= 0:
            raise ValueError(
                "self_distillation.teacher_update_interval must be a positive integer "
                f"when set, got {self.teacher_update_interval}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.teacher_prompt_mode is not None and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                f"self_distillation.teacher_prompt_mode must be None or 'answer_hint', got {self.teacher_prompt_mode}"
            )
        if self.teacher_always_on and not self.teacher_image_key and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                "self_distillation.teacher_image_key is required when teacher_always_on=True "
                "(unless teacher_prompt_mode='answer_hint')"
            )
        if self.counterfactual_null_mode not in (None, "mean_color", "cut"):
            raise ValueError(
                "self_distillation.counterfactual_null_mode must be None, 'mean_color' or 'cut', "
                f"got {self.counterfactual_null_mode}"
            )
        if self.counterfactual_null_mode == "cut":
            if not self.teacher_always_on:
                raise ValueError("counterfactual_null_mode='cut' requires teacher_always_on=True.")
            if not (self.teacher_model_source == "fixed" or
                    (self.teacher_model_source == "legacy" and self.teacher_regularization == "frozen")):
                raise ValueError("counterfactual_null_mode='cut' requires a fixed or frozen teacher.")
        if self.counterfactual_beta_gate not in ("none", "verifier"):
            raise ValueError(
                f"self_distillation.counterfactual_beta_gate must be 'none' or 'verifier', got {self.counterfactual_beta_gate}"
            )
        if self.counterfactual_beta_gate != "none" and self.counterfactual_null_mode is None:
            raise ValueError("counterfactual_beta_gate requires counterfactual_null_mode.")
        if self.counterfactual_gate_on_scale < 0.0 or self.counterfactual_gate_off_scale < 0.0:
            raise ValueError("counterfactual_gate_on_scale and counterfactual_gate_off_scale must be non-negative.")
        if int(self.skip_first_tokens) < 0:
            raise ValueError("self_distillation.skip_first_tokens must be non-negative.")
        if self.counterfactual_extrapolation_beta < 0.0:
            raise ValueError(
                "self_distillation.counterfactual_extrapolation_beta must be non-negative, "
                f"got {self.counterfactual_extrapolation_beta}"
            )
        if self.counterfactual_null_mode is not None and not self.full_logit_distillation:
            raise ValueError("Visual-counterfactual target reconstruction requires full_logit_distillation=True.")
        valid_teacher_model_source = ["legacy", "current", "fixed"]
        if self.teacher_model_source not in valid_teacher_model_source:
            raise ValueError(
                "self_distillation.teacher_model_source must be one of "
                f"{valid_teacher_model_source}, got {self.teacher_model_source}"
            )
        if self.teacher_model_source == "fixed" and not self.teacher_model_path:
            raise ValueError("self_distillation.teacher_model_path is required when teacher_model_source='fixed'")
        if self.teacher_regularization == "progressive":
            if self.teacher_model_source != "legacy":
                raise ValueError(
                    "self_distillation.teacher_regularization='progressive' requires "
                    "teacher_model_source='legacy'"
                )
            if self.teacher_update_interval is None:
                raise ValueError(
                    "self_distillation.teacher_update_interval is required when "
                    "teacher_regularization='progressive'"
                )


@dataclass
class RouterReplayConfig(BaseConfig):
    pass

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        pass
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    pass

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    pass

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False

    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)


    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        pass
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        pass
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        pass
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    pass

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        pass
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    pass

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        pass
        super().__post_init__()
        self.engine = self.fsdp_config


        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        pass
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
