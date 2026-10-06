"""Generic hook eligibility and dispatch without vLLM or a GPU."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_lens import Hook, SteeringVector


@pytest.fixture
def worker(monkeypatch):
    forward_context = ModuleType("vllm.forward_context")
    forward_context.is_forward_context_available = lambda: True
    forward_context.get_forward_context = lambda: None
    model_utils = ModuleType("vllm.model_executor.models.utils")
    model_utils.PPMissingLayer = type("PPMissingLayer", (), {})
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)
    path = Path(__file__).resolve().parents[2] / "vllm_lens" / "_worker_ext.py"
    spec = importlib.util.spec_from_file_location("worker_hook_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class QueryStarts:
    """Track metadata reads that would synchronize a CUDA tensor."""

    def __init__(self, values, allowed=True):
        self.values = values
        self.allowed = allowed
        self.reads = 0

    def __getitem__(self, index):
        def tolist():
            assert self.allowed, "Unexpected boundary read"
            self.reads += 1
            return self.values[index]

        return SimpleNamespace(tolist=tolist)


def make_extension(worker, monkeypatch, starts, *, persistent=(), per_request=((),)):
    req_ids = [f"r{i}-internal" for i in range(len(per_request))]
    requests = {
        req_id: SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))
        for req_id in req_ids
    }
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(req_ids), req_ids=req_ids),
        requests=requests,
        model=object(),
    )
    # vLLM keeps one forward context for every layer of a forward pass.
    forward_context = SimpleNamespace(
        attn_metadata={"attention": SimpleNamespace(query_start_loc=starts)}
    )
    monkeypatch.setattr(worker, "get_forward_context", lambda: forward_context)
    return SimpleNamespace(
        model_runner=runner,
        _persistent_hooks=list(persistent),
        _hook_data={f"r{i}": list(hooks) for i, hooks in enumerate(per_request)},
        _steering_data={},
        _hook_contexts={},
        _persistent_hook_contexts={},
        _prefetched_params={},
        _captured_states={},
    )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "pre,hook_pre,hook_layer",
    [(False, False, 4), (False, True, 3), (True, True, 4), (True, False, 3)],
)
def test_ineligible_hooks_do_not_clone_or_read_boundaries(
    worker, monkeypatch, persistent, pre, hook_pre, hook_layer
):
    hook = Hook(
        fn=lambda *_: pytest.fail("Inactive hook ran"),
        layer_indices=[hook_layer],
        pre=hook_pre,
    )
    starts = QueryStarts([0, 2], allowed=False)
    extension = make_extension(
        worker,
        monkeypatch,
        starts,
        persistent=[hook] if persistent else [],
        per_request=[[] if persistent else [hook]],
    )
    hidden = torch.ones(2, 3)
    monkeypatch.setattr(
        torch.Tensor, "clone", lambda *_: pytest.fail("Inactive hook cloned tensor")
    )
    dispatch = worker._pre_hook_inner if pre else worker._hook_inner
    assert dispatch(extension, 3, hidden) is None
    assert starts.reads == 0
    assert extension._hook_contexts == extension._persistent_hook_contexts == {}


@pytest.mark.parametrize("pre", [False, True])
def test_only_eligible_request_reads_boundaries(worker, monkeypatch, pre):
    hooks = [
        Hook(fn=lambda *_: pytest.fail("Wrong layer"), layer_indices=[4], pre=pre),
        Hook(fn=lambda _, h: h + 5, layer_indices=[3], pre=pre),
        Hook(
            fn=lambda *_: pytest.fail("Wrong category"),
            layer_indices=[3],
            pre=not pre,
        ),
    ]
    starts = QueryStarts([0, 2, 3, 6])
    extension = make_extension(
        worker, monkeypatch, starts, per_request=[[hook] for hook in hooks]
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    expected = hidden.clone()
    expected[2:3] += 5
    dispatch = worker._pre_hook_inner if pre else worker._hook_inner
    result = dispatch(extension, 3, hidden)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(hidden, torch.arange(6).reshape(6, 1).float())
    assert starts.reads == 1
    assert set(extension._hook_contexts) == {"r1-internal"}
    assert extension._hook_contexts["r1-internal"][0].seq_len == 1


@pytest.mark.parametrize("fused", [False, True])
def test_mixed_hooks_keep_indices_and_compose_in_category_order(
    worker, monkeypatch, fused
):
    def add(ctx, h):
        ctx.saved["seen"] = h.clone()
        return h + 1

    def multiply(ctx, h):
        ctx.saved["seen"] = h.clone()
        return h * 2

    def mixed_hooks(fn):
        return [
            Hook(fn=lambda *_: pytest.fail("Wrong layer"), layer_indices=[4]),
            Hook(fn=fn, layer_indices=[3], pre=True),
            Hook(fn=fn, layer_indices=[3]),
        ]

    starts = QueryStarts([0, 2])
    extension = make_extension(
        worker,
        monkeypatch,
        starts,
        persistent=mixed_hooks(add),
        per_request=[mixed_hooks(multiply)],
    )
    hidden = torch.tensor([[1.0], [3.0]])
    pre_result = worker._pre_hook_inner(extension, 3, hidden)
    torch.testing.assert_close(pre_result, (hidden + 1) * 2)
    output = (hidden, torch.full_like(hidden, 10)) if fused else hidden
    stream = output[0] + output[1] if fused else output
    post_result = worker._hook_inner(extension, 3, output)
    post_stream = post_result[0] + post_result[1] if fused else post_result
    torch.testing.assert_close(post_stream, (stream + 1) * 2)
    for store in (extension._hook_contexts, extension._persistent_hook_contexts):
        assert set(store["r0-internal"]) == {1, 2}
    persistent = extension._persistent_hook_contexts["r0-internal"]
    per_request = extension._hook_contexts["r0-internal"]
    torch.testing.assert_close(persistent[1].saved["seen"], hidden)
    torch.testing.assert_close(per_request[1].saved["seen"], hidden + 1)
    torch.testing.assert_close(persistent[2].saved["seen"], stream)
    torch.testing.assert_close(per_request[2].saved["seen"], stream + 1)


@pytest.mark.parametrize("steering", [False, True])
def test_inactive_hooks_preserve_capture_and_steering(worker, monkeypatch, steering):
    hook = Hook(fn=lambda *_: pytest.fail("Wrong layer"), layer_indices=[4])
    starts = QueryStarts([0, 2])
    extension = make_extension(worker, monkeypatch, starts, persistent=[hook])
    req_id = "r0-internal"
    extension.model_runner.requests[req_id].sampling_params.extra_args = {
        "output_residual_stream": [3]
    }
    if steering:
        extension._steering_data["r0"] = [
            SteeringVector(activations=torch.ones(1, 2), layer_indices=[3])
        ]
    hidden = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    result = worker._hook_inner(extension, 3, hidden)
    expected = hidden + 1 if steering else hidden
    captured = extension._captured_states[req_id][3][0]
    torch.testing.assert_close(captured, expected)
    if steering:
        torch.testing.assert_close(result, expected)
    else:
        assert result is None
    assert extension._persistent_hook_contexts == {}


def test_one_plan_and_boundary_read_per_forward_pass(worker, monkeypatch):
    seen = []
    hooks = [
        Hook(
            fn=lambda ctx, h: seen.append(("pre", ctx.layer_idx)),
            layer_indices=[0, 1],
            pre=True,
        ),
        Hook(
            fn=lambda ctx, h: seen.append(("post", ctx.layer_idx)), layer_indices=[0, 1]
        ),
    ]
    starts = QueryStarts([0, 2, 3])
    extension = make_extension(worker, monkeypatch, starts, per_request=[hooks, []])
    extension.model_runner.requests["r1-internal"].sampling_params.extra_args = {
        "output_residual_stream": "[1]"
    }
    lookups = []
    find = worker._find_hook_configs_no_persistent
    monkeypatch.setattr(
        worker,
        "_find_hook_configs_no_persistent",
        lambda ext, req_id, extra: lookups.append(req_id) or find(ext, req_id, extra),
    )
    hidden = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    for layer in range(4):
        worker._pre_hook_inner(extension, layer, hidden)
        worker._hook_inner(extension, layer, hidden)

    assert lookups == ["r0-internal", "r1-internal"]
    assert starts.reads == 1
    assert seen == [("pre", 0), ("post", 0), ("pre", 1), ("post", 1)]
    assert list(extension._captured_states) == ["r1-internal"]
    torch.testing.assert_close(
        extension._captured_states["r1-internal"][1][0], hidden[2:]
    )


def test_new_forward_pass_rebuilds_plan(worker, monkeypatch):
    starts = QueryStarts([0, 2])
    extension = make_extension(worker, monkeypatch, starts)
    hidden = torch.ones(2, 1)
    assert worker._hook_inner(extension, 0, hidden) is None
    # Registrations change between scheduler steps, with a new forward context.
    extension._hook_data["r0"] = [Hook(fn=lambda _, h: h * 3, layer_indices=[0])]
    next_context = SimpleNamespace(
        attn_metadata={"attention": SimpleNamespace(query_start_loc=starts)}
    )
    monkeypatch.setattr(worker, "get_forward_context", lambda: next_context)
    torch.testing.assert_close(worker._hook_inner(extension, 0, hidden), hidden * 3)


def test_steering_for_other_layers_skips_clone_and_boundaries(worker, monkeypatch):
    starts = QueryStarts([0, 2], allowed=False)
    extension = make_extension(worker, monkeypatch, starts)
    extension._steering_data["r0"] = [
        SteeringVector(activations=torch.ones(1, 2), layer_indices=[5])
    ]
    monkeypatch.setattr(
        torch.Tensor, "clone", lambda *_: pytest.fail("Inactive steering cloned")
    )
    assert worker._hook_inner(extension, 3, torch.ones(2, 2)) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ([1, 2], frozenset({1, 2})),
        ("[1, 2]", frozenset({1, 2})),
        (True, True),
        ("not json", True),
    ],
)
def test_capture_layers_parsing(worker, value, expected):
    extra = {} if value is None else {"output_residual_stream": value}
    assert worker._capture_layers(extra) == expected
