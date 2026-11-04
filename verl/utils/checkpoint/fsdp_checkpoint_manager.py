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

import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    pass

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    pass

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        pass
        if local_path is None:
            return


        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        saved_world_size = self._saved_world_size(local_path)
        if saved_world_size is not None and saved_world_size != self.world_size:
            self._load_resharded(local_path, saved_world_size)
            return


        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

            if "rng" in extra_state_dict:

                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )


        torch.distributed.barrier()

    def _saved_world_size(self, local_path):
        import glob
        import re

        if is_non_local(local_path) or not os.path.isdir(local_path):
            return None
        if os.path.exists(os.path.join(local_path, f"model_world_size_{self.world_size}_rank_0.pt")):
            return None
        sizes = set()
        for path in glob.glob(os.path.join(local_path, "*_world_size_*_rank_0.pt")):
            match = re.search(r"_world_size_(\d+)_rank_0\.pt$", path)
            if match:
                sizes.add(int(match.group(1)))
        if len(sizes) > 1:
            raise ValueError(f"checkpoint {local_path} mixes files saved with world sizes {sorted(sizes)}")
        return sizes.pop() if sizes else None

    def _load_resharded(self, local_path, old_world_size):
        import torch.nn.functional as F
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.tensor import DTensor, Shard

        if fsdp_version(self.model) != 1:
            raise NotImplementedError("resharded resume is only implemented for FSDP1 checkpoints")
        W, N, rank = old_world_size, self.world_size, self.rank
        log_with_rank(f"[reshard] checkpoint saved with {W} ranks, loading into {N} ranks: {local_path}",
                      rank=rank, logger=logger)
        if rank == 0:
            print(f"[reshard] checkpoint saved with {W} ranks, loading into {N} ranks: {local_path}", flush=True)

        def shard_file(kind, r):
            path = os.path.join(local_path, f"{kind}_world_size_{W}_rank_{r}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"resharded resume needs every saved rank file, missing {path}")
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

        if self.should_load_model:
            shards = [shard_file("model", r) for r in range(W)]
            full = {}
            for key, ref in shards[0].items():
                if isinstance(ref, DTensor):
                    if len(ref.placements) != 1 or not isinstance(ref.placements[0], Shard) or ref.placements[0].dim != 0:
                        raise ValueError(f"unexpected placement {ref.placements} for {key}")
                    tensor = torch.cat([s[key].to_local() for s in shards], dim=0)
                    if tuple(tensor.shape) != tuple(ref.shape):
                        raise ValueError(f"{key}: rebuilt shape {tuple(tensor.shape)} != saved global shape {tuple(ref.shape)}")
                    full[key] = tensor
                else:
                    full[key] = ref
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                self.model.load_state_dict(full)
            del shards, full

        if self.should_load_optimizer:
            olds = [shard_file("optim", r) for r in range(W)]
            params = [p for group in self.optimizer.param_groups for p in group["params"]]
            saved_ids = [i for group in olds[0]["param_groups"] for i in group["params"]]
            if len(saved_ids) != len(params) or saved_ids != list(range(len(params))):
                raise ValueError(f"optimizer layout differs: saved {len(saved_ids)} params, current {len(params)}")
            state = {}
            for idx, param in enumerate(params):
                if idx not in olds[0]["state"]:
                    continue
                unpadded = param._unpadded_unsharded_size.numel()
                shard = param._sharded_size.numel()
                if shard * N != param._padded_unsharded_size.numel():
                    raise ValueError(f"param {idx}: shard {shard} x {N} != padded {param._padded_unsharded_size.numel()}")
                entry = {}
                for name, value in olds[0]["state"][idx].items():
                    if torch.is_tensor(value) and value.dim() == 1:
                        parts = [o["state"][idx][name] for o in olds]
                        old_shard = parts[0].numel()
                        if any(x.numel() != old_shard for x in parts) or not 0 <= W * old_shard - unpadded < W:
                            raise ValueError(f"param {idx} {name}: {W} shards of {old_shard} do not cover {unpadded} elements")
                        flat = torch.cat(parts)[:unpadded]
                        flat = F.pad(flat, (0, shard * N - unpadded))
                        entry[name] = flat[rank * shard:(rank + 1) * shard].clone()
                    else:
                        entry[name] = value.clone() if torch.is_tensor(value) else value
                state[idx] = entry
            self.optimizer.load_state_dict({"state": state, "param_groups": olds[0]["param_groups"]})
            del olds, state

        if self.should_load_extra:
            extra = torch.load(os.path.join(local_path, f"extra_state_world_size_{W}_rank_{rank % W}.pt"),
                               map_location="cpu", weights_only=False)
            if "rng" in extra:
                self.load_rng_state(extra["rng"])
            if extra.get("lr_scheduler") is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        torch.distributed.barrier()
        if rank == 0:
            print(f"[reshard] loaded {local_path} ({W} -> {N} ranks)", flush=True)
        roundtrip = os.environ.get("VERL_RESHARD_ROUNDTRIP_DIR")
        if roundtrip:
            self.save_checkpoint(roundtrip, global_step=-1)
            self.previous_saved_paths = [p for p in self.previous_saved_paths
                                         if os.path.abspath(p) != os.path.abspath(roundtrip)]
            torch.distributed.barrier()
            if rank == 0:
                print(f"[reshard] round-trip copy written to {roundtrip}", flush=True)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        pass
        if local_path is None:
            return


        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()


        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )


        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:


            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:


                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:

                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )


            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)


            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)


        torch.distributed.barrier()

        if self.should_save_hf_model:


            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:

                    import transformers
                    from packaging import version

                    if version.parse(transformers.__version__) >= version.parse("4.54.0"):

                        from transformers import AutoModelForImageTextToText

                        auto_model_cls = AutoModelForImageTextToText
                    else:

                        from transformers import AutoModelForVision2Seq

                        auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model


            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
