"""Read-only probe execution, shared by batched and callback-order paths."""

from __future__ import annotations

from typing import Any

import torch

from vllm_lens._helpers.types import HookContext, LinearProbe


def save_probe_scores(
    probe: LinearProbe, context: HookContext, hidden_states: torch.Tensor
) -> None:
    """Evaluate one request when arbitrary hooks require request-major order."""
    with torch.no_grad():
        scores = (hidden_states.float() @ probe.weights.T).cpu()
    context.saved.setdefault(f"L{context.layer_idx}", []).append(scores)


def run_batched_probes(
    probes: list[tuple[int, LinearProbe]],
    layer_idx: int,
    hidden_states: torch.Tensor,
    runner: Any,
    query_start_loc: torch.Tensor,
    store: dict[str, dict[int, HookContext]],
) -> None:
    """Project a packed forward once per bank, then split reduced CPU scores.

    vLLM 0.30's model runner mirrors boundaries in a ``CpuGpuBuffer``.
    Use its CPU mirror only when metadata starts at the same device buffer;
    sliced or alternate metadata falls back to one combined transfer.
    """
    num_reqs = runner.input_batch.num_reqs
    runner_boundaries = getattr(runner, "query_start_loc", None)
    cpu_boundaries: Any = getattr(runner_boundaries, "cpu", None)
    gpu_boundaries: Any = getattr(runner_boundaries, "gpu", None)
    if not (
        isinstance(cpu_boundaries, torch.Tensor)
        and cpu_boundaries.device.type == "cpu"
        and isinstance(gpu_boundaries, torch.Tensor)
        and isinstance(query_start_loc, torch.Tensor)
        and gpu_boundaries.device == query_start_loc.device
        and gpu_boundaries.data_ptr() == query_start_loc.data_ptr()
    ):
        cpu_boundaries = query_start_loc.detach().cpu()
    if cpu_boundaries.ndim != 1 or cpu_boundaries.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("query_start_loc must be a one-dimensional integer tensor")
    boundaries = cpu_boundaries[: num_reqs + 1].tolist()
    if (
        len(boundaries) != num_reqs + 1
        or boundaries[0] != 0
        or boundaries[-1] > hidden_states.shape[0]
        or any(start > end for start, end in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("Invalid query_start_loc for linear probes")
    total_tokens = boundaries[-1]
    if total_tokens == 0:
        return

    with torch.no_grad():
        # One FP32 conversion is shared by all banks. Only reduced scores
        # leave the device, never the full residual stream.
        packed_hidden = hidden_states[:total_tokens].float()
        for position, probe in probes:
            scores = (packed_hidden @ probe.weights.T).cpu()
            for req_id, start, end in zip(
                runner.input_batch.req_ids, boundaries, boundaries[1:]
            ):
                if start == end:
                    continue
                contexts = store.setdefault(req_id, {})
                context = contexts.get(position)
                if context is None:
                    context = HookContext()
                    contexts[position] = context
                context.layer_idx = layer_idx
                context.seq_len = end - start
                # Clone CPU slices so one result does not retain the whole
                # scheduler batch or share storage with another request.
                context.saved.setdefault(f"L{layer_idx}", []).append(
                    scores[start:end].clone()
                )
