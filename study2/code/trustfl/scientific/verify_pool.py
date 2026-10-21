"""Frozen v3 pool-only verification: <=70 backward calls, zero optimizers.

Never invokes a scientific round, dataset preparation, native CUDA backward,
create_graph, or a training entry point. Missing numerical capabilities are
BLOCKED. Tolerances and fixture inventory live in the immutable registry.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

COUNTS = {"pool_fixtures": 12, "model_fixtures": 2, "guards": 6,
          "max_pool_backward": 60, "max_model_backward": 10,
          "max_total_backward": 70}
ORDER = ("native_cpu_reference", "repaired_cpu_client", "repaired_cpu_root",
         "repaired_cuda_client", "repaired_cuda_root")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot(root):
    paths = [*(root / "trustfl").rglob("*.py"),
             *(root / "tests" / "scientific").glob("*"),
             *(root / "scripts" / "scientific_transfer").glob("*"),
             *(root / "scripts" / "scientific_pool_transfer").glob("*")]
    return {str(path.relative_to(root)): {
        "raw": digest_file(path),
        "lf": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
        for path in sorted(paths) if path.is_file()}


def fixture_result(case_id, kind, status, **details):
    return {"case_id": case_id, "kind": kind, "status": status, **details}


class BackwardBudget:
    """Counts attempted calls before backward and completed calls afterwards."""
    def __init__(self):
        self.counts = {kind: {"attempted": 0, "completed": 0}
                       for kind in ("pool", "model", "total")}
        self.keys = set()

    def backward(self, kind, key, output, upstream):
        limit = COUNTS["max_" + kind + "_backward"]
        if key in self.keys or self.counts[kind]["attempted"] >= limit or self.counts["total"]["attempted"] >= 70:
            raise RuntimeError("Frozen backward budget exceeded or duplicate evaluation")
        self.keys.add(key)
        self.counts[kind]["attempted"] += 1
        self.counts["total"]["attempted"] += 1
        output.backward(upstream)
        self.counts[kind]["completed"] += 1
        self.counts["total"]["completed"] += 1


@contextmanager
def optimizer_prohibition(torch):
    audit = {"optimizer_construction_attempts": 0, "optimizer_steps": 0,
             "enforcement": "Torch absent; optimizer execution unavailable"}
    if torch is None:
        yield audit
        return
    old = torch.optim.Optimizer.__init__
    def forbid(*args, **kwargs):
        audit["optimizer_construction_attempts"] += 1
        raise RuntimeError("Frozen v3 verification prohibits optimizer construction")
    torch.optim.Optimizer.__init__ = forbid
    audit["enforcement"] = "Scoped optimizer base constructor prohibition; no preexisting optimizer"
    try:
        yield audit
    finally:
        torch.optim.Optimizer.__init__ = old


@contextmanager
def deterministic_scope(torch):
    if torch is None:
        yield {}
        return
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield {"algorithms": torch.are_deterministic_algorithms_enabled(),
               "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]


def tensor_hash(value):
    data = value.detach().cpu().contiguous().numpy()
    header = json.dumps([str(data.dtype), list(data.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + data.tobytes()).hexdigest()


def state_hash(state):
    hashes = {key: tensor_hash(value) for key, value in sorted(state.items())}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def compare_array(actual, expected, atol, rtol):
    """Nonfinite locations/signs are exact; finite error is reported separately."""
    import numpy as np
    a = actual.detach().cpu().numpy() if hasattr(actual, "detach") else np.asarray(actual)
    b = expected.detach().cpu().numpy() if hasattr(expected, "detach") else np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"passed": False, "reason": "shape/dtype mismatch"}
    masks_equal = (np.array_equal(np.isnan(a), np.isnan(b)) and
                   np.array_equal(np.isposinf(a), np.isposinf(b)) and
                   np.array_equal(np.isneginf(a), np.isneginf(b)))
    finite = np.isfinite(a) & np.isfinite(b)
    difference = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    ref = np.abs(b[finite].astype(np.float64))
    bound = atol + rtol * ref
    finite_ok = bool(np.all(difference <= bound))
    scaled = None if np.any((bound == 0) & (difference > 0)) else (float(np.divide(difference, bound, out=np.zeros_like(difference), where=bound > 0).max()) if difference.size else 0.0)
    return {"passed": bool(masks_equal and finite_ok), "nonfinite_patterns_equal": bool(masks_equal),
            "max_abs_error": float(difference.max()) if difference.size else 0.0,
            "max_scaled_error": scaled,
            "nonfinite_count": int((~np.isfinite(b)).sum()), "atol": atol, "rtol": rtol}


def independent_reference(x, upstream, native_indices):
    """Independent CPU oracle: explicit adaptive bins and indexed scalar adds.

    It neither imports the repaired implementation nor calls its helper. NumPy
    argmax independently checks finite-input bin routing. Gradient accumulation
    uses the native saved indices, one output bin at a time, and one distinct
    input element per plane; there is no parallel repeated-index accumulation.
    """
    import numpy as np
    a = x.detach().cpu().numpy()
    g = upstream.detach().cpu().numpy()
    indices = native_indices.detach().cpu().numpy()
    n, c, h, w = a.shape
    out = np.empty((n, c, 7, 7), dtype=a.dtype)
    independent_indices = np.empty((n, c, 7, 7), dtype=np.int64)
    for row in range(7):
        start_h, end_h = (row * h) // 7, ((row + 1) * h + 6) // 7
        for col in range(7):
            start_w, end_w = (col * w) // 7, ((col + 1) * w + 6) // 7
            window = a[:, :, start_h:end_h, start_w:end_w]
            flat = window.reshape(n, c, -1)
            chosen = flat.argmax(axis=-1)
            independent_indices[:, :, row, col] = (start_h + chosen // (end_w - start_w)) * w + start_w + chosen % (end_w - start_w)
            out[:, :, row, col] = np.take_along_axis(flat, chosen[:, :, None], axis=-1)[:, :, 0]
    grad = np.zeros((n * c, h * w), dtype=a.dtype)
    idx = indices.reshape(n * c, 49)
    values = g.reshape(n * c, 49)
    planes = np.arange(n * c)
    with np.errstate(invalid="ignore", over="ignore"):
        for position in range(49):
            grad[planes, idx[:, position]] = grad[planes, idx[:, position]] + values[:, position]
    return out, independent_indices, grad.reshape(a.shape)


def pool_inputs(spec, torch):
    generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
    dtype = getattr(torch, spec["dtype"])
    x = torch.randn(spec["shape"], generator=generator, dtype=dtype)
    g = torch.randn((*spec["shape"][:2], 7, 7), generator=generator, dtype=dtype)
    kind = spec["kind"]
    if kind == "all_zero_ties": x.zero_()
    elif kind == "all_equal_negative": x.fill_(-3.25)
    elif kind == "repeated_plateaus":
        row = torch.arange(16)[:, None] // 3
        col = torch.arange(16)[None, :] // 3
        x.copy_(((row + col) % 3).to(dtype).expand_as(x))
    elif kind in ("overlap_peak", "cancellation_upstream"):
        x.fill_(-8)
        x[:, :, 9, 9] = 100
        if kind == "cancellation_upstream":
            g.zero_()
            g[:, :, 3, 3], g[:, :, 3, 4] = 1e8, 1
            g[:, :, 4, 3], g[:, :, 4, 4] = -1e8, 3
    elif kind == "negative_values": x = -x.abs() - 0.125
    elif kind == "mixed_sign_upstream":
        g = (torch.arange(g.numel()).reshape_as(g) % 7 - 3).to(dtype) / 3
    elif kind == "zero_upstream": g.zero_()
    elif kind == "noncontiguous": x = x.transpose(-1, -2)
    elif kind == "nonfinite_upstream_isolation":
        # Nonfinite *upstream* only: no runtime rejection policy is introduced.
        g.zero_(); g[:, :, 0, 0] = float("nan"); g[:, :, 6, 6] = float("inf")
    return x, g


def evaluation_unavailable(torch, device):
    if torch is None: return "Torch dependency unavailable"
    if importlib.util.find_spec("numpy") is None: return "NumPy dependency unavailable"
    if device == "cuda" and not torch.cuda.is_available(): return "CUDA unavailable; CPU cannot substitute"
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        return "CUDA verification requires scoped CUBLAS_WORKSPACE_CONFIG before Python starts"
    return None


def run_pool(spec, registry, torch, budget):
    evaluations = []
    cached = {}
    source = upstream = None
    tolerance = registry["pool_tolerances"][spec["dtype"]]
    for label in ORDER:
        device = "cuda" if "cuda" in label else "cpu"
        reason = evaluation_unavailable(torch, device)
        if reason:
            evaluations.append({"evaluation": label, "status": "BLOCKED", "reason": reason})
            continue
        try:
            if source is None: source, upstream = pool_inputs(spec, torch)
            if label != ORDER[0] and "reference" not in cached:
                raise RuntimeError("Native CPU reference failed; no authenticated comparison target")
            from .pooling import adaptive_max_pool2d_fixed_order
            x = source.to(device).detach().requires_grad_(True)
            if spec["kind"] == "noncontiguous" and x.is_contiguous():
                raise AssertionError("P10 must exercise an actually noncontiguous input on each device")
            grad = upstream.to(device)
            if label == ORDER[0]:
                out, indices = torch.nn.functional.adaptive_max_pool2d(x, (7, 7), return_indices=True)
            else:
                out, indices = adaptive_max_pool2d_fixed_order(x, return_indices=True)
            if indices.requires_grad or indices.dtype != torch.int64:
                raise AssertionError("Saved indices must be nondifferentiable int64")
            budget.backward("pool", spec["id"] + "/" + label, out, grad)
            current = {"out": out.detach().cpu(), "indices": indices.detach().cpu(), "grad": x.grad.detach().cpu()}
            hashes = {key: tensor_hash(value) for key, value in current.items()}
            if label == ORDER[0]:
                oracle_out, oracle_idx, oracle_grad = independent_reference(source, upstream, current["indices"])
                checks = {"independent_forward": compare_array(current["out"], oracle_out, 0, 0),
                          "independent_indices": compare_array(current["indices"], oracle_idx, 0, 0),
                          "independent_gradient": compare_array(current["grad"], oracle_grad, **tolerance)}
                cached["reference"] = current
                cached["oracle_gradient"] = oracle_grad
            else:
                reference = cached["reference"]
                checks = {"native_cpu_forward": compare_array(current["out"], reference["out"], 0, 0),
                          "native_cpu_indices": compare_array(current["indices"], reference["indices"], 0, 0),
                          "native_cpu_gradient": compare_array(current["grad"], reference["grad"], **tolerance),
                          "independent_gradient": compare_array(current["grad"], cached["oracle_gradient"], **tolerance)}
                if device in cached:
                    checks["repeat_bitwise"] = {"passed": hashes == cached[device], "first": cached[device], "second": hashes}
                else:
                    cached[device] = hashes
            evaluations.append({"evaluation": label, "status": "PASS" if all(v["passed"] for v in checks.values()) else "FAIL",
                                "hashes": hashes, "checks": checks, "input_stride": list(x.stride()),
                                "input_contiguous": x.is_contiguous(), "backward_completed": True})
        except Exception as exc:
            evaluations.append({"evaluation": label, "status": "FAIL", "error": repr(exc)})
    status = aggregate_status(evaluations)
    return fixture_result(spec["id"], spec["kind"], status, evaluations=evaluations,
                          input_hash=tensor_hash(source) if source is not None else None,
                          upstream_hash=tensor_hash(upstream) if upstream is not None else None)


def aggregate_status(entries):
    if any(entry["status"] == "FAIL" for entry in entries): return "FAIL"
    if any(entry["status"] == "BLOCKED" for entry in entries): return "BLOCKED"
    return "PASS"


def run_model(spec, registry, torch, budget):
    evaluations, cached = [], {}
    native = repaired = source = upstream = None
    identities = {}
    for label in ORDER:
        device = "cuda" if "cuda" in label else "cpu"
        reason = evaluation_unavailable(torch, device)
        if reason:
            evaluations.append({"evaluation": label, "status": "BLOCKED", "reason": reason})
            continue
        try:
            if native is None:
                from ..models import CNNEuroSAT
                from .models import build_model, ScientificCNNEuroSAT
                torch.manual_seed(spec["seed"])
                native = CNNEuroSAT()
                native_rng = torch.get_rng_state().clone()
                torch.manual_seed(spec["seed"])
                repaired = build_model(SimpleNamespace(name=spec["factory_name"], num_classes=10), "eurosat")
                repaired_rng = torch.get_rng_state().clone()
                if not isinstance(repaired, ScientificCNNEuroSAT): raise AssertionError("Factory did not choose repaired EuroSAT")
                if len(list(native.parameters())) != len(list(repaired.parameters())): raise AssertionError("Parameter keys changed")
                if sum(p.numel() for p in repaired.parameters()) != 1664970: raise AssertionError("Parameter count changed")
                if list(native.state_dict()) != list(repaired.state_dict()): raise AssertionError("State dictionary keys changed")
                for key, value in native.state_dict().items():
                    other = repaired.state_dict()[key]
                    if value.shape != other.shape or not torch.equal(value, other): raise AssertionError("Seeded parameter identity changed: " + key)
                if not torch.equal(native_rng, repaired_rng): raise AssertionError("Initialization RNG consumption changed")
                if any(type(getattr(native, key)) is not type(getattr(repaired, key)) or repr(getattr(native, key)) != repr(getattr(repaired, key))
                       for key in ("conv1", "conv2", "fc1", "fc2")):
                    raise AssertionError("Surrounding layer identity changed")
                strict = repaired.load_state_dict(native.state_dict(), strict=True)
                if strict.missing_keys or strict.unexpected_keys: raise AssertionError("Strict state loading failed")
                identities = {"state_sha256": state_hash(native.state_dict()), "parameter_count": 1664970,
                              "initialization_tensors_equal": True, "rng_consumption_equal": True,
                              "strict_state_load": True, "surrounding_layers_equal": True,
                              "factory": type(repaired).__name__, "state_keys": list(repaired.state_dict())}
                generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + 100)
                source = torch.randn(spec["shape"], generator=generator)
                if spec["kind"] == "zero_input_ties": source.zero_()
                upstream = torch.randn((spec["shape"][0], 10), generator=generator)
            if label != ORDER[0] and "reference" not in cached:
                raise RuntimeError("Native CPU full-model reference failed")
            model = copy.deepcopy(native if label == ORDER[0] else repaired).to(device)
            if label != ORDER[0] and model.__class__ is not repaired.__class__: raise AssertionError("Deepcopy lost repaired class")
            model.load_state_dict(native.state_dict(), strict=True)
            if state_hash(model.state_dict()) != identities["state_sha256"]: raise AssertionError("Device/deepcopy parameter state differs")
            x = source.to(device).detach().requires_grad_(True)
            out = model(x)
            budget.backward("model", spec["id"] + "/" + label, out, upstream.to(device))
            current = {"output": out.detach().cpu(), "input_gradient": x.grad.detach().cpu()}
            for key, parameter in model.named_parameters():
                if parameter.grad is None: raise AssertionError("Missing parameter gradient: " + key)
                current["parameter_gradient/" + key] = parameter.grad.detach().cpu()
            if not all(torch.isfinite(value).all().item() for value in current.values()): raise AssertionError("Nonfinite whole-model fixture output/gradient")
            hashes = {key: tensor_hash(value) for key, value in current.items()}
            checks = {}
            if label == ORDER[0]:
                cached["reference"] = current
            else:
                checks = {key: compare_array(value, cached["reference"][key], **registry["model_cpu_tolerances" if device == "cpu" else "model_tolerances"])
                          for key, value in current.items()}
                if device in cached:
                    checks["repeat_bitwise"] = {"passed": cached[device] == hashes, "first": cached[device], "second": hashes}
                else: cached[device] = hashes
            evaluations.append({"evaluation": label, "status": "PASS" if all(v["passed"] for v in checks.values()) else "FAIL",
                                "hashes": hashes, "checks": checks, "backward_completed": True,
                                "form": "native" if label == ORDER[0] else "deepcopy_client" if label.endswith("client") else "deepcopy_root"})
        except Exception as exc:
            evaluations.append({"evaluation": label, "status": "FAIL", "error": repr(exc)})
    return fixture_result(spec["id"], spec["kind"], aggregate_status(evaluations), evaluations=evaluations, identities=identities)


def read_ast(root, relative):
    return ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)


def run_guards(root, registry, torch, before):
    results = []
    for spec in registry["guards"]:
        try:
            case = spec["id"]
            details = {"backward_calls": 0, "scope": "static/stdlib"}
            if case == "G01":
                ids = [x["id"] for group in ("pool", "model", "guards") for x in registry[group]]
                assert registry["planned_counts"] == COUNTS
                assert len(set(ids)) == len(ids) == 20
                assert [x["id"] for x in registry["pool"]] == [f"P{i:02d}" for i in range(1, 13)]
                assert [x["id"] for x in registry["model"]] == ["M01", "M02"]
                assert [x["id"] for x in registry["guards"]] == [f"G{i:02d}" for i in range(1, 7)]
                assert registry["evaluation_order"] == list(ORDER)
                assert len(registry["pool"]) * len(ORDER) == 60
                assert len(registry["model"]) * len(ORDER) == 10
                assert all(x["backward_calls"] == 0 for x in registry["guards"])
            elif case == "G02":
                native = (root / "trustfl/models.py").read_bytes().replace(b"\r\n", b"\n")
                assert hashlib.sha256(native).hexdigest() == registry["native_models_sha256_lf"]
            elif case == "G03":
                relative = "trustfl/scientific/scenario.py"
                tree = read_ast(root, relative)
                imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and any(alias.name == "build_model" for alias in node.names)]
                assert len(imports) == 2 and all(node.module == "models" and node.level == 1 for node in imports), relative
                details[relative] = len(imports)
                run_tree = read_ast(root, "trustfl/scientific/run.py")
                builder = next(node for node in run_tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_components")
                assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "preflight_tensors" for node in ast.walk(builder))
                assert any(isinstance(node, ast.ImportFrom) and node.module == "scenario" and node.level == 1 and any(alias.name == "preflight_tensors" for alias in node.names) for node in ast.walk(builder))
                assert not any(isinstance(node, ast.ImportFrom) and node.level > 1 and node.module == "models" for node in ast.walk(run_tree))
                details["build_components"] = "calls verified scientific preflight_tensors"
                tree = read_ast(root, "trustfl/scientific/server.py")
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "deepcopy"]
                assert len(calls) >= 2
                model_tree = read_ast(root, "trustfl/scientific/models.py")
                subclass = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "ScientificCNNEuroSAT")
                assert not any(isinstance(node, ast.FunctionDef) and node.name == "__init__" for node in subclass.body)
            elif case == "G04":
                tree = read_ast(root, "trustfl/scientific/pooling.py")
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                names = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
                assert not any(any(term in name.lower() for term in ("scatter", "index_add", "atomic")) for name in names)
                assert not any(name in ("Optimizer", "SGD", "Adam", "AdamW", "step", "use_deterministic_algorithms") for name in names)
                assert "where" in names and "adaptive_max_pool2d" in names
                assert any(isinstance(node, ast.For) for node in ast.walk(tree))
                details["limitation"] = "Structural checks complement, and do not establish, numerical equivalence"
            elif case == "G05":
                assert before == source_snapshot(root)
                text = (root / "trustfl/scientific/pooling.py").read_text()
                assert "is_grad_enabled" in text and "backward" in text
                tree = read_ast(root, "trustfl/scientific/pooling.py")
                backwards = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "backward"]
                assert backwards
                assert any(isinstance(node, ast.Raise) for node in ast.walk(backwards[0]))
                details["higher_order"] = "Static refusal contract inspected; no extra create_graph/backward probe"
            elif case == "G06":
                if torch is None:
                    results.append(fixture_result(case, spec["kind"], "BLOCKED", reason="Torch unavailable; unsupported-input forward guards unexecuted", backward_calls=0))
                    continue
                from .pooling import adaptive_max_pool2d_fixed_order
                invalid = [("spatial", torch.zeros((1, 1, 15, 16))),
                           ("rank", torch.zeros((1, 16, 16))),
                           ("dtype", torch.zeros((1, 1, 16, 16), dtype=torch.int64)),
                           ("half", torch.zeros((1, 1, 16, 16), dtype=torch.float16)),
                           ("empty_batch", torch.zeros((0, 1, 16, 16)))]
                if [name for name, _ in invalid] != spec["invalid_input_subcases"]:
                    raise AssertionError("Frozen unsupported-input subcase registry changed")
                rejected = []
                for name, tensor in invalid:
                    try: adaptive_max_pool2d_fixed_order(tensor)
                    except (TypeError, ValueError, RuntimeError) as exc: rejected.append({"case": name, "exception": type(exc).__name__})
                    else: raise AssertionError("Unsupported input silently accepted: " + name)
                details.update({"scope": "forward guards only", "rejected": rejected})
            results.append(fixture_result(case, spec["kind"], "PASS", **details))
        except Exception as exc:
            results.append(fixture_result(spec["id"], spec["kind"], "FAIL", error=repr(exc), backward_calls=0))
    return results


def verify(root, out):
    registry_path = root / "tests/scientific/pool_fixture_registry.json"
    registry = json.loads(registry_path.read_text())
    before = source_snapshot(root)
    write_json(out / "source-before.json", before)
    write_json(out / "frozen-registry.json", registry)
    torch = None
    dependency_error = None
    if importlib.util.find_spec("torch") is not None:
        try: import torch
        except Exception as exc: dependency_error = repr(exc)
    budget = BackwardBudget()
    started = time.perf_counter()
    # Validate immutable inventory before any numeric call, not after spending a budget.
    if (registry["planned_counts"] != COUNTS or registry["evaluation_order"] != list(ORDER) or
            [x["id"] for x in registry["pool"]] != [f"P{i:02d}" for i in range(1, 13)] or
            [x["id"] for x in registry["model"]] != ["M01", "M02"] or
            [x["id"] for x in registry["guards"]] != [f"G{i:02d}" for i in range(1, 7)]):
        raise ValueError("Frozen v3 fixture registry differs from reviewed inventory")
    with optimizer_prohibition(torch) as audit, deterministic_scope(torch) as precision:
        fixtures = [run_pool(spec, registry, torch, budget) for spec in registry["pool"]]
        fixtures += [run_model(spec, registry, torch, budget) for spec in registry["model"]]
        fixtures += run_guards(root, registry, torch, before)
    after = source_snapshot(root)
    write_json(out / "source-after.json", after)
    preserved = before == after
    status = aggregate_status(fixtures)
    if not preserved or audit["optimizer_construction_attempts"]:
        status = "FAIL"
    if status == "PASS" and (budget.counts["total"] != {"attempted": 70, "completed": 70} or len(fixtures) != 20):
        status = "FAIL"
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "trustfl" or name.startswith("trustfl."):
            filename = getattr(module, "__file__", None)
            if filename:
                location = Path(filename).resolve()
                origins[name] = str(location)
                if not location.is_relative_to(root): status = "FAIL"
    report = {"schema": "scientific-pool-verification-v3", "status": status,
              "registry_sha256": digest_file(registry_path), "planned_counts": COUNTS,
              "backward_counts": budget.counts, "optimizer_steps": audit["optimizer_steps"],
              "optimizer_construction_attempts": audit["optimizer_construction_attempts"],
              "optimizer_guard": audit, "source_before_after_equal": preserved,
              "fixtures": fixtures, "precision_settings": precision, "import_origins": origins,
              "seconds": time.perf_counter() - started, "integrated_attempts_executed": 0,
              "native_cuda_backward_executed": False, "training_authorized": False,
              "claim_eligible": False, "dependencies": {"torch": torch is not None, "torch_import_error": dependency_error},
              "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                              "torch": str(torch.__version__) if torch is not None else None,
                              "cuda_runtime": str(torch.version.cuda) if torch is not None else None,
                              "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
                              "gpu": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None}}
    write_json(out / "report.json", report)
    return 0 if status == "PASS" else 3 if status == "BLOCKED" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--assert-zero-optimizer-steps", action="store_true", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if root != Path(args.expected_source).resolve() or Path.cwd().resolve() != root or os.environ.get("PYTHONPATH"):
        parser.error("Explicit candidate cwd/import root required; PYTHONPATH overrides refused")
    from ..comparison.provenance import safe_new_target
    out = safe_new_target(args.output, protected=[root])
    out.mkdir(exist_ok=False)
    try:
        return verify(root, out)
    except BaseException as exc:
        write_json(out / "suite-failure.json", {"status": "FAIL", "error": repr(exc), "claim_eligible": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
