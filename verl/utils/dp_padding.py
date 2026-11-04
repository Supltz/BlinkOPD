import torch

CAPABILITY_VERSION = 2


def ceil_div(value, divisor):
    if value < 0 or divisor < 1:
        raise ValueError("Invalid batch size or DP size")
    return (value + divisor - 1) // divisor


def padding_needed(batch, dp_size):
    return len(batch) % dp_size != 0


def pad_inference_batch(batch, dp_size):
    from verl.protocol import pad_dataproto_to_divisor
    result, count = pad_dataproto_to_divisor(batch, dp_size)
    if count:
        flags = torch.zeros(len(result), dtype=torch.bool, device=result.batch["responses"].device)
        flags[-count:] = True
        result.batch["gpu_infer_pad"] = flags
    return result, count


def unpad_inference_batch(output, count):
    from verl.protocol import unpad_dataproto
    result = unpad_dataproto(output, count)
    if result.batch is not None and "gpu_infer_pad" in result.batch:
        result.batch.pop("gpu_infer_pad")
    return result


def pad_actor_batch(batch, dp_size, nominal_rows=None):
    from verl.protocol import pad_dataproto_to_divisor
    result = batch.select_idxs(list(range(len(batch))))
    result.meta_info = dict(batch.meta_info)
    real_rows = len(result)
    result, count = pad_dataproto_to_divisor(result, dp_size)
    result.meta_info = dict(result.meta_info)
    flags = torch.zeros(len(result), dtype=torch.bool, device=result.batch["response_mask"].device)
    if count:
        flags[-count:] = True
    result.batch["gpu_pad"] = flags
    result.batch["response_mask"] = result.batch["response_mask"] * (~flags).unsqueeze(1)
    if "iso_keep_len" in result.batch:
        result.batch["iso_keep_len"] = result.batch["iso_keep_len"] * (~flags)
    result.meta_info.update(gpu_pad_rows=count, gpu_pad_real_rows=real_rows,
                            gpu_pad_loss_rows=nominal_rows or real_rows,
                            gpu_pad_dp_size=dp_size, gpu_pad_prepared=True)
    return result


def real_micro_batch(batch, pad_key="gpu_pad"):
    if pad_key not in batch.batch:
        return batch
    real = (~batch.batch[pad_key].bool()).nonzero(as_tuple=False).flatten()
    if real.numel():
        return batch.select_idxs(real)
    return batch.select_idxs([0])


def score_micro_batch(actor, batch, temperature, calculate_entropy, pad_token_id):
    flags = batch.batch.get("gpu_infer_pad")
    scored = real_micro_batch(batch, "gpu_infer_pad")
    model_inputs = {**scored.batch, **scored.non_tensor_batch, "pad_token_id": pad_token_id}
    with torch.no_grad():
        outputs = actor._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=calculate_entropy)
    return restore_padded_outputs(outputs, flags)


def restore_padded_outputs(outputs, flags):
    if flags is None:
        return outputs
    original_rows = flags.shape[0]
    scored_rows = max(1, int((~flags.bool()).sum().item()))
    restored = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == scored_rows:
            result = value.new_zeros((original_rows,) + value.shape[1:])
            real = (~flags.bool()).to(value.device)
            if real.any():
                result[real] = value
            restored[key] = result
        else:
            restored[key] = value
    return restored


def microbatch_loss_scale(response_mask, global_real_rows, dp_size, gpu_pad=None):
    if global_real_rows < 1 or dp_size < 1:
        raise ValueError("The logical mini-batch and DP size must be positive")
    real = response_mask.shape[0] if gpu_pad is None else int((~gpu_pad.bool()).sum().item())
    return real * dp_size / global_real_rows


def real_row_mean(values, flags=None):
    if flags is None:
        return values.mean()
    real = (~flags.bool()).to(values.device)
    return values[real].sum() / real.sum().clamp_min(1)


def update_actor_with_padding(trainer, batch):
    from verl.protocol import DataProto
    from verl.utils.metric import reduce_metrics
    actor = trainer.config.actor_rollout_ref.actor
    dp_size = trainer._get_dp_size(trainer.actor_rollout_wg, "actor")
    nominal = actor.ppo_mini_batch_size * trainer.config.actor_rollout_ref.rollout.n
    if trainer.use_legacy_worker_impl == "disable":
        raise ValueError("Ragged DP batches require the legacy FSDP actor used by the project run scripts")
    mini_batches = batch.split(nominal)
    outputs = []
    total = len(mini_batches) * actor.ppo_epochs
    for epoch in range(actor.ppo_epochs):
        for index, mini in enumerate(mini_batches):
            payload = pad_actor_batch(mini, dp_size, nominal)
            payload.meta_info.update(gpu_pad_epoch_once=True,
                                    gpu_pad_force_off_policy=total != 1,
                                    gpu_pad_defer_teacher_update=epoch * len(mini_batches) + index + 1 < total)
            metrics = {}
            if trainer.config.trainer.balance_batch:
                trainer._balance_batch(payload, metrics)
            print(f"[gpu-pad] step={trainer.global_steps} mini={index + 1}/{len(mini_batches)} "
                  f"real={len(mini)} transport={len(payload)} dp={dp_size}", flush=True)
            output = trainer._update_actor(payload)
            outputs.append(reduce_metrics(output.meta_info["metrics"]))
    combined = {key: [entry[key] for entry in outputs] for key in outputs[0]}
    return DataProto.from_single_dict(data={}, meta_info={"metrics": combined})
