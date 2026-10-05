"""Probe math, hook ordering, and existing transports without vLLM or a GPU."""

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cloudpickle
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm_lens import Hook, LinearProbe, SteeringVector, deserialize_hook_results
from vllm_lens._activations_plugin import _decode_hooks, _llm_collect_hook_results
from vllm_lens._helpers._linear_probe import run_batched_probes
from vllm_lens._helpers._serialize import merge_persistent_hook_results
from vllm_lens._helpers.types import parse_hook
from vllm_lens._hooks_router import router
from vllm_lens.client import VLLMLensClient


@pytest.fixture
def worker(monkeypatch):
    forward = ModuleType("vllm.forward_context")
    forward.is_forward_context_available = lambda: True
    forward.get_forward_context = lambda: None
    utils = ModuleType("vllm.model_executor.models.utils")
    utils.PPMissingLayer = type("PPMissingLayer", (), {})
    monkeypatch.setitem(sys.modules, forward.__name__, forward)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = Path(__file__).resolve().parents[2] / "vllm_lens" / "_worker_ext.py"
    spec = importlib.util.spec_from_file_location("worker_linear_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_extension(
    worker, monkeypatch, *, persistent=(), per_request=((), ()), lengths=(2, 1)
):
    req_ids = [f"r{i}-internal" for i in range(len(per_request))]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(req_ids), req_ids=req_ids),
        requests={
            r: SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))
            for r in req_ids
        },
        query_start_loc=SimpleNamespace(cpu=starts, gpu=starts),
        device=torch.device("cpu"),
        model=torch.nn.Linear(3, 3),
    )
    monkeypatch.setattr(
        worker,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "attention": SimpleNamespace(query_start_loc=runner.query_start_loc.gpu)
            }
        ),
    )
    extension = worker.HiddenStatesExtension()
    extension.model_runner = runner
    extension.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 8, get_hidden_size=lambda: 3
    )
    extension._persistent_hooks = list(persistent)
    extension._hook_data = {f"r{i}": list(h) for i, h in enumerate(per_request)}
    extension._steering_data = {}
    extension._hook_contexts = {}
    extension._persistent_hook_contexts = {}
    extension._prefetched_params = {}
    extension._captured_states = {}
    extension._should_capture = True
    return extension


def test_probe_snapshots_cpu_fp32_weights_and_roundtrips():
    source = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    probe = LinearProbe(weights=source, layer_indices=[3, 4])
    source.detach().zero_()
    assert probe.weights.dtype == torch.float32
    assert not probe.weights.requires_grad and probe.weights.device.type == "cpu"
    assert torch.equal(probe.weights, torch.ones(2, 3))
    decoded = _decode_hooks(json.dumps([probe.model_dump()]))[0]
    assert isinstance(decoded, LinearProbe) and decoded.has_layer(4)
    assert torch.equal(decoded.weights, probe.weights)
    assert isinstance(cloudpickle.loads(cloudpickle.dumps(probe)), LinearProbe)
    assert parse_hook(probe) is probe


@pytest.mark.parametrize(
    "weights",
    [
        torch.ones(3),
        torch.ones(1, 2, 3),
        torch.ones(0, 3),
        torch.ones(2, 0),
        torch.ones(2, 3, dtype=torch.int64),
        "weights",
    ],
)
def test_probe_rejects_invalid_weights(weights):
    with pytest.raises(ValidationError, match="weights"):
        LinearProbe(weights=weights, layer_indices=[3])


def test_probe_rejects_pre_hooks_and_empty_layers():
    with pytest.raises(ValidationError, match="pre"):
        LinearProbe(weights=torch.ones(2, 3), layer_indices=[3], pre=True)
    with pytest.raises(ValidationError, match="layer_indices"):
        LinearProbe(weights=torch.ones(2, 3), layer_indices=[])


@pytest.mark.parametrize("fused", [False, True])
def test_persistent_probe_batches_once_and_preserves_outputs(
    worker, monkeypatch, fused
):
    probe = LinearProbe(
        weights=torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.0]]), layer_indices=[3]
    )
    # Pre-only callbacks do not interfere with the post-layer batch path.
    pre = Hook(
        fn=lambda *_: pytest.fail("Pre-hook ran as post-hook"),
        layer_indices=[3],
        pre=True,
    )
    extension = make_extension(
        worker, monkeypatch, persistent=[pre, probe], per_request=[[pre], []]
    )
    hidden = torch.arange(12).reshape(4, 3).to(torch.bfloat16)
    output = (hidden, torch.ones_like(hidden)) if fused else hidden
    expected = hidden[:3] + 1 if fused else hidden[:3]
    original = hidden.clone()
    calls = []
    matmul = torch.Tensor.__matmul__
    monkeypatch.setattr(
        torch.Tensor,
        "__matmul__",
        lambda a, b: calls.append((a.shape, b.shape)) or matmul(a, b),
    )
    # Trusted runner CPU metadata means GPU scalar boundary reads are unnecessary.
    monkeypatch.setattr(
        torch.Tensor, "item", lambda _: pytest.fail("GPU scalar boundary read")
    )
    assert worker._hook_inner(extension, 3, output) is None
    assert calls == [(torch.Size([3, 3]), torch.Size([3, 2]))]
    for req_id, sl in [("r0-internal", slice(0, 2)), ("r1-internal", slice(2, 3))]:
        context = extension._persistent_hook_contexts[req_id][1]
        torch.testing.assert_close(
            context.saved["L3"][0], matmul(expected[sl].float(), probe.weights.T)
        )
        assert context.seq_len == len(expected[sl])
        assert context.saved["L3"][0].device.type == "cpu"
    assert torch.equal(hidden, original)
    assert extension._hook_contexts == {}


def test_probe_chunks_follow_requests_when_batch_order_changes(worker, monkeypatch):
    probe = LinearProbe(weights=torch.eye(3), layer_indices=[3, 4])
    extension = make_extension(worker, monkeypatch, persistent=[probe])
    first = torch.arange(9).reshape(3, 3).float()
    worker._hook_inner(extension, 3, first)
    runner = extension.model_runner
    runner.input_batch.req_ids.reverse()
    runner.query_start_loc.cpu = torch.tensor([0, 1, 2])
    runner.query_start_loc.gpu = runner.query_start_loc.cpu
    second = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    worker._hook_inner(extension, 3, second)
    worker._hook_inner(extension, 4, second)
    r0 = extension._persistent_hook_contexts["r0-internal"][0].saved
    r1 = extension._persistent_hook_contexts["r1-internal"][0].saved
    torch.testing.assert_close(torch.cat(r0["L3"]), torch.cat([first[:2], second[1:]]))
    torch.testing.assert_close(torch.cat(r1["L3"]), torch.cat([first[2:], second[:1]]))
    torch.testing.assert_close(r0["L4"][0], second[1:])
    # One request's chunk owns only its own reduced storage.
    assert r0["L3"][0].untyped_storage().nbytes() == 2 * 3 * 4


@pytest.mark.parametrize("fused", [False, True])
def test_mixed_callbacks_preserve_probe_mutation_probe_order(
    worker, monkeypatch, fused
):
    events = []
    probe = LinearProbe(weights=torch.eye(3), layer_indices=[3])

    def modify(ctx, hidden):
        events.append(float(hidden[0, 0]))
        return hidden + 5

    modifier = Hook(fn=modify, layer_indices=[3])
    per_request = Hook(
        fn=lambda ctx, h: ctx.saved.update(seen=h.clone()), layer_indices=[3]
    )
    extension = make_extension(
        worker,
        monkeypatch,
        persistent=[probe, modifier, probe],
        per_request=[[per_request], [per_request]],
    )
    hidden = torch.arange(9).reshape(3, 3).float()
    output = (hidden, torch.ones_like(hidden)) if fused else hidden
    initial = hidden + 1 if fused else hidden.clone()
    result = worker._hook_inner(extension, 3, output)
    final = result[0] + result[1] if fused else result
    torch.testing.assert_close(final, initial + 5)
    assert events == [float(initial[0, 0]), float(initial[2, 0])]
    for req_id, sl in [("r0-internal", slice(0, 2)), ("r1-internal", slice(2, 3))]:
        contexts = extension._persistent_hook_contexts[req_id]
        torch.testing.assert_close(contexts[0].saved["L3"][0], initial[sl])
        torch.testing.assert_close(contexts[2].saved["L3"][0], initial[sl] + 5)
        torch.testing.assert_close(
            extension._hook_contexts[req_id][0].saved["seen"], initial[sl] + 5
        )
    torch.testing.assert_close(hidden, torch.arange(9).reshape(3, 3).float())


def test_probe_sees_steering_and_preserves_unrelated_requests(worker, monkeypatch):
    probe = LinearProbe(weights=torch.eye(3), layer_indices=[3])
    extension = make_extension(worker, monkeypatch, per_request=[[probe], []])
    extension._steering_data["r0"] = [
        SteeringVector(activations=torch.ones(1, 3), layer_indices=[3])
    ]
    hidden = torch.arange(9).reshape(3, 3).float()
    result = worker._hook_inner(extension, 3, hidden)
    torch.testing.assert_close(
        extension._hook_contexts["r0-internal"][0].saved["L3"][0], hidden[:2] + 1
    )
    torch.testing.assert_close(result[2:], hidden[2:])
    assert "r1-internal" not in extension._hook_contexts


def test_non_capture_rank_skips_probes_but_runs_mutating_callbacks(worker, monkeypatch):
    probe = LinearProbe(weights=torch.ones(1, 3), layer_indices=[3])
    extension = make_extension(
        worker,
        monkeypatch,
        persistent=[probe, Hook(fn=lambda _, h: h + 1, layer_indices=[3])],
    )
    extension._should_capture = False
    result = worker._hook_inner(extension, 3, torch.zeros(3, 3))
    assert torch.equal(result, torch.ones(3, 3))
    assert all(
        0 not in contexts for contexts in extension._persistent_hook_contexts.values()
    )


def test_boundary_fallback_transfers_once_and_skips_empty_requests(monkeypatch):
    probe = LinearProbe(weights=torch.eye(3), layer_indices=[3])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=3, req_ids=["a", "empty", "b"])
    )
    starts = torch.tensor([0, 2, 2, 3])
    store = {}
    cpu = torch.Tensor.cpu
    transfers = []
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda t: transfers.append(t.shape) or cpu(t)
    )
    run_batched_probes([(4, probe)], 3, torch.ones(3, 3), runner, starts, store)
    assert transfers == [torch.Size([4]), torch.Size([3, 3])]
    assert set(store) == {"a", "b"} and set(store["a"]) == {4}


def test_offset_metadata_view_does_not_use_full_runner_cpu_mirror(monkeypatch):
    probe = LinearProbe(weights=torch.eye(3), layer_indices=[3])
    gpu_buffer = torch.tensor([999, 0, 2, 3])
    metadata = gpu_buffer[1:]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["a", "b"]),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 1, 3]), gpu=gpu_buffer),
    )
    hidden = torch.arange(9).reshape(3, 3).float()
    store = {}
    cpu = torch.Tensor.cpu
    transfers = []
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda t: transfers.append(t.shape) or cpu(t)
    )
    run_batched_probes([(0, probe)], 3, hidden, runner, metadata, store)
    assert transfers == [torch.Size([3]), torch.Size([3, 3])]
    torch.testing.assert_close(store["a"][0].saved["L3"][0], hidden[:2])
    torch.testing.assert_close(store["b"][0].saved["L3"][0], hidden[2:])


def test_empty_batch_does_not_project():
    runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0, req_ids=[]))
    store = {}
    run_batched_probes(
        [(0, LinearProbe(weights=torch.eye(3), layer_indices=[3]))],
        3,
        torch.empty(0, 3),
        runner,
        torch.tensor([0]),
        store,
    )
    assert store == {}


def test_registration_owns_weights_and_clear_results_keeps_registration(
    worker, monkeypatch
):
    extension = make_extension(worker, monkeypatch)
    monkeypatch.setattr(extension, "install_hooks", lambda: None)
    probe = LinearProbe(weights=torch.ones(2, 3), layer_indices=[3])
    extension.set_persistent_hooks(cloudpickle.dumps([probe]))
    registered = extension._persistent_hooks[0]
    assert registered.weights.data_ptr() != probe.weights.data_ptr()
    worker._hook_inner(extension, 3, torch.ones(3, 3))
    assert extension.get_all_hook_results() is not None
    extension.clear_persistent_hook_results()
    assert extension._persistent_hooks == [registered]
    assert extension.get_all_hook_results() is None
    extension.clear_persistent_hooks()
    assert extension._persistent_hooks == []
    for invalid in [
        LinearProbe(weights=torch.ones(2, 4), layer_indices=[3]),
        LinearProbe(weights=torch.ones(2, 3), layer_indices=[8]),
    ]:
        with pytest.raises(ValueError):
            extension.set_persistent_hooks(cloudpickle.dumps([invalid]))
    assert extension._persistent_hooks == []


def test_offline_and_http_merge_probe_results_without_tp_duplicates():
    first = {"r": {"1": {"L3": [torch.ones(2, 4)]}}}
    second = {"r": {"1": {"L7": [torch.zeros(1, 4)]}}}
    first["r"]["0"] = {"generic": ["stage0-rank0"]}
    second["r"]["0"] = {"generic": ["stage1-rank0"]}
    replica0 = {"r": {"0": {"generic": ["stage0-rank1"]}}}
    replica1 = {"r": {"0": {"generic": ["stage1-rank1"]}}}
    raw = [
        pickle.dumps(first),
        pickle.dumps(replica0),
        pickle.dumps(second),
        pickle.dumps(replica1),
    ]
    engine = SimpleNamespace(
        collective_rpc=lambda _: raw,
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(tensor_parallel_size=2)
            )
        ),
    )
    merged = _llm_collect_hook_results(engine)
    assert set(merged["r"]["1"]) == {"L3", "L7"}
    assert len(merged["r"]["1"]["L3"]) == 1
    assert merged["r"]["0"]["generic"] == ["stage0-rank0", "stage1-rank0"]
    torch.testing.assert_close(merged["r"]["1"]["L3"][0], first["r"]["1"]["L3"][0])
    assert merge_persistent_hook_results([]) == {}

    calls = []

    async def rpc(method, args=()):
        calls.append((method, args))
        return raw if method == "get_all_hook_results" else None

    app = FastAPI()
    app.state.engine_client = SimpleNamespace(collective_rpc=rpc)
    app.include_router(router)
    client = TestClient(app)
    probe = LinearProbe(weights=torch.ones(4, 3), layer_indices=[3, 7])
    assert (
        client.post(
            "/v1/hooks/register", json={"hooks": [probe.model_dump()]}
        ).status_code
        == 200
    )
    registered = cloudpickle.loads(calls[0][1][0])[0]
    assert isinstance(registered, LinearProbe)
    collected = deserialize_hook_results(
        client.post("/v1/hooks/collect").json()["results"]["r"]
    )
    assert set(collected["1"]) == {"L3", "L7"}
    assert collected["0"]["generic"] == [
        "stage0-rank0",
        "stage0-rank1",
        "stage1-rank0",
        "stage1-rank1",
    ]
    torch.testing.assert_close(collected["1"]["L7"][0], second["r"]["1"]["L7"][0])


def test_http_client_sends_probe_discriminator(monkeypatch):
    client = VLLMLensClient("http://localhost", model="test")
    calls = []
    monkeypatch.setattr(
        client._session,
        "post",
        lambda url, **kw: (
            calls.append((url, kw)) or SimpleNamespace(json=lambda: {"status": "ok"})
        ),
    )
    probe = LinearProbe(weights=torch.ones(2, 3), layer_indices=[3])
    client.register_hooks([probe])
    encoded = calls[0][1]["json"]["hooks"][0]
    assert encoded["kind"] == "linear_probe"
    assert isinstance(parse_hook(encoded), LinearProbe)
