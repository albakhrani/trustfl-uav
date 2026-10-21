"""Manual one-scenario/one-attempt scientific adapter; never an automatic grid.

Preparing a bundle initializes data/model only. The separate run action can
perform SGD and requires an explicit later training-authorization acknowledgement.
This module has no import-time Torch dependency or execution side effect.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .protocol import METHODS, POLICIES, DATASETS, DATASET_SPECS, make_protocol, load_protocol, native_config_dict
from .records import AttemptWriter, write_json_once, validate_attempt, _failure_audit_value
from .instrumentation import StepBudget, MemorySampler
from .precision import (precision_identity, require_precision_identity,
                        precision_policy_scope, require_precision_active)


def _source_root():
    return Path(__file__).resolve().parents[2]


def _protection(original_root, scenario=None, data_root=None, protected_roots=()):
    original=Path(original_root)
    if not original.is_absolute() or not original.is_dir():
        raise ValueError("Existing absolute preserved original-root is required")
    original=original.resolve(); source=_source_root()
    if source==original or source.is_relative_to(original):
        raise ValueError("Scientific adapter must execute in a separate isolated candidate")
    protected=[original,source]
    if scenario is not None:protected.append(Path(scenario).resolve())
    if data_root is not None:protected.append(Path(data_root).resolve())
    # Previously verified Windows v1 roots are protected even if a caller omits
    # optional extra roots. Their existing contents are never used as outputs.
    protected.extend(original.parent/name for name in (
        "trustfl-uav-common-policy-v1","trustfl-uav-native-reference-v1","common-policy-verification-v1"))
    prior_mac=Path("/private/tmp/trustfl-common-policy-implementation-2026-10-03-v1")
    if prior_mac.is_dir():protected.append(prior_mac)
    for path in protected_roots:
        path=Path(path)
        if not path.is_absolute():raise ValueError("Protected roots must be absolute")
        protected.append(path.resolve())
    return protected


def assert_runtime_imports():
    """Verify actual module origins, not HEAD alone, before any execution."""
    source=_source_root()
    if Path.cwd().resolve()!=source or os.environ.get("PYTHONPATH"):
        raise RuntimeError("Use isolated candidate cwd with no PYTHONPATH override")
    names=("trustfl","trustfl.scientific.run","trustfl.scientific.server","trustfl.scientific.context",
           "trustfl.scientific.datasets","trustfl.scientific.scenario","trustfl.scientific.methods",
           "trustfl.scientific.records","trustfl.scientific.analysis","trustfl.fl.client",
           "trustfl.fl.server","trustfl.trust.trust","trustfl.trust.selection",
           "trustfl.attacks.attacks","trustfl.uav.network","trustfl.models",
           "trustfl.scientific.models","trustfl.scientific.pooling","trustfl.scientific.precision")
    origins={}
    for name in names:
        module=importlib.import_module(name)
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(source/"trustfl"):
            raise RuntimeError("Imported-source conflict: "+name+" -> "+str(path))
        raw=path.read_bytes()
        origins[name]={"path":str(path),"sha256_raw":hashlib.sha256(raw).hexdigest(),
                       "sha256_lf_normalized":hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()}
    return origins


def _maximum_steps(bundle, method):
    cfg=bundle["resolved_config"]; train=cfg["train"]
    counts=bundle["indices"]["counts"]
    client_per_epoch=sorted((math.ceil(n/train["batch_size"]) for n in counts["clients"]),reverse=True)
    client=train["rounds"]*train["local_epochs"]*sum(client_per_epoch[:cfg["trust"]["k_max"]])
    root=(train["rounds"]*train["local_epochs"]*math.ceil(counts["root"]/256)
          if method=="fltrust_declared_root_fallback" else 0)
    return {"client":client,"root":root,"total":client+root,
            "scope":"conservative fixed-horizon upper bound; unavailable clients may reduce realized work"}


def build_components(bundle,scenario,method,policy,device,writer,step_budget,host_limit_bytes):
    """Construct prospective state; this function performs no optimizer step."""
    require_precision_active(bundle.get("precision_execution_identity"))
    import numpy as np
    from ..config import ExperimentConfig
    from ..uav.network import UAVNetwork
    from ..attacks.attacks import AttackController
    from ..trust.trust import TrustManager
    from .scenario import preflight_tensors,load_arrays
    from .server import ScientificClient,ScientificServer
    from .context import ScientificContext
    cfg=ExperimentConfig.from_dict(native_config_dict(bundle["protocol"],bundle["resolved_config"]["data"]["data_dir"],method))
    cfg.train.device=device
    index_arrays,trace_arrays=load_arrays(scenario)
    model,(specifications,root,test)=preflight_tensors(bundle,scenario,check_content=False)
    context=ScientificContext(bundle,trace_arrays,cfg,writer,method,policy,step_budget,host_limit_bytes)
    clients=[ScientificClient(uid,base,indices,cfg.train,device,cfg.seed,context=context)
             for uid,(base,indices) in enumerate(specifications)]
    network=UAVNetwork(cfg.uav,cfg.seed,context=context)
    for uid,state in enumerate(bundle["initial_uav"]):
        if state["uid"]!=uid:raise ValueError("Initial client UID order mismatch")
        for key,value in state.items():
            setattr(network.uavs[uid],key,np.asarray(value,dtype=float) if key in ("pos","vel") else copy.deepcopy(value))
    network._gm_mean={int(key):np.asarray(value,dtype=float) for key,value in bundle["gm_mean"].items()}
    attack=AttackController(cfg.attack,cfg.uav.num_uavs,cfg.seed,context=context)
    attack.malicious=set(bundle["memberships"]["malicious"])
    attack._adaptive_attacking={uid:False for uid in attack.malicious}
    attack._sys_mode={uid:None for uid in attack.malicious}
    attack._sys_rotation={uid:0 for uid in attack.malicious}
    attack.apply_static_effects(network)
    classes=cfg.model.num_classes
    labels=index_arrays["train_labels"]
    # This is the unchanged native label_variance formula, now applied to
    # unambiguous ultimate IDs rather than ambiguous nested Subset positions.
    sigma2={}
    for uid,(_,indices) in enumerate(specifications):
        histogram=np.bincount(labels[indices],minlength=classes)/len(indices)
        sigma2[uid]=float(np.var(histogram)*classes**2)
    server=ScientificServer(cfg,model,clients,network,TrustManager(cfg.trust,cfg.uav.num_uavs),
                            attack,root,test,sigma2,device,str(writer.path),context=context)
    return server,context


def _physical_memory_bytes():
    if sys.platform=="win32":
        from .windows_memory import physical_memory_bytes
        return physical_memory_bytes()
    try:return int(os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
    except (ValueError,OSError,AttributeError) as exc:
        raise RuntimeError("Physical RAM query unavailable; cannot enforce declared75%ceiling") from exc


def _execution_environment(device):
    require_precision_active(precision_identity())
    import torch
    import torchvision
    from ..comparison.provenance import environment
    value=environment();value["torchvision"]=torchvision.__version__
    value["selected_device"]=device;value["host_physical_ram_bytes"]=_physical_memory_bytes()
    value["torch_num_threads"]=torch.get_num_threads()
    value["torch_num_interop_threads"]=torch.get_num_interop_threads()
    value["cudnn_version"]=torch.backends.cudnn.version()
    if device=="cuda":
        index=torch.cuda.current_device();prop=torch.cuda.get_device_properties(index)
        value["cuda_selected"]={"logical_index":index,"name":prop.name,"total_memory_bytes":prop.total_memory,
                                  "compute_capability":[prop.major,prop.minor],
                                  "uuid":str(prop.uuid) if getattr(prop,"uuid",None) is not None else None}
    else:value["cuda_selected"]=None
    return value


def _device_gate(device):
    """Availability gate only; precision mutation belongs to the shared scope.

    The old separately gated V6 caller may use this read-only gate before
    scenario preparation. This neither enters execution nor authorizes training.
    """
    import torch
    if device not in ("cpu","cuda"):raise ValueError("Explicit cpu/cuda only; no fallback")
    if device=="cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Exact reviewed CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no CPU fallback")


def _classify_failure(exc):
    from .context import ScientificFailure
    if isinstance(exc,ScientificFailure):return exc.kind,dict(exc.evidence)
    if isinstance(exc,(KeyboardInterrupt,SystemExit)):
        return "external_interruption",{"interruption_recorded":True}
    # An explicit own-process ceiling check is attributable to this attempt.
    # A generic CUDA allocator OOM may involve outside users and is ambiguous.
    if isinstance(exc,MemoryError) and str(exc) in (
        "declared host memory ceiling exceeded","declared model GPU allocation ceiling exceeded"):
        return "budget_oom",{"oom_exception_recorded":True,"declared_resource_budget":True,
                              "same_declared_execution_device":True,"own_allocation_ceiling_recorded":True}
    if isinstance(exc,(ValueError,FileNotFoundError)):
        return "missing_corrupt_evidence",{"evidence_integrity_failure_recorded":True}
    return "ambiguous_cause",{}


def run_entry(scenario,method,policy,device,output,*,step_budget,host_limit_bytes,
              original_root,protected_roots=(),training_authorized=False,fullrun_adapter=None):
    """One explicit future attempt. No caller gets implicit training consent.

    A shared StepBudget is supported for the separately authorized V6 sequence.
    This callable does not create a grid, replace a failed seed or resume files.
    """
    if training_authorized is not True:
        raise PermissionError("Scientific execution requires separate explicit training authorization")
    if method not in METHODS or policy not in POLICIES:
        raise ValueError("Unsupported scientific method/policy")
    if not isinstance(step_budget,StepBudget):raise TypeError("Explicit StepBudget required")
    if type(host_limit_bytes) is not int or not 0<host_limit_bytes<=24*1024**3:
        raise ValueError("Explicit positive host memory ceiling no larger than24GiBrequired")
    from ..comparison.provenance import safe_new_target,source_identity,environment,tensor_state_hash
    from .scenario import load_bundle
    from .datasets import inspect_offline
    import torch
    imports=assert_runtime_imports()
    bundle=load_bundle(scenario,check_source=True)
    require_precision_identity(bundle.get("precision_execution_identity"))
    is_v5 = bundle["protocol"]["purpose"] == "v5"
    if is_v5 != (fullrun_adapter is not None):
        raise ValueError("v5 requires its full-run adapter; other profiles reject it")
    v5_metadata = fullrun_adapter.validate(bundle,method,policy,step_budget,scenario) if is_v5 else {}
    if device!=bundle["protocol"]["device"]:
        raise ValueError("Execution device differs from sealed scenario configuration")
    if bundle["protocol"]["purpose"]=="v6" and policy!="common":
        raise ValueError("V6 is common-policy only")
    protected=_protection(original_root,scenario,bundle["resolved_config"]["data"]["data_dir"],protected_roots)
    path=safe_new_target(output,protected=protected)
    observed=inspect_offline(bundle["protocol"]["dataset"],bundle["resolved_config"]["data"]["data_dir"])
    if observed["manifest"]!=bundle["dataset_manifest"]:
        raise ValueError("Dataset content changed since immutable scenario preparation")
    del observed
    physical_ram=_physical_memory_bytes()
    if host_limit_bytes>min(24*1024**3,int(0.75*physical_ram)):
        raise ValueError("Host memory ceiling exceeds min24GiB,75%actual physicalRAM")
    bound=_maximum_steps(bundle,method)
    if step_budget.maximum-step_budget.reserved<bound["total"]:
        raise ValueError("Explicit remaining optimizer budget is below conservative per-attempt maximum")
    cfg=bundle["resolved_config"]; source=source_identity(_source_root())
    starting_budget=copy.deepcopy(step_budget.snapshot())
    metadata={"record_contract":"scientific-production-v2","planned_rounds":cfg["train"]["rounds"],
              "protocol_id":bundle["protocol_hash"],"scenario_id":bundle["bundle_id"],
              "source_identity":source,"model_execution_identity":bundle["model_execution_identity"],
              "precision_execution_identity":bundle["precision_execution_identity"],
              "claim_eligible":False,"client_ids":list(range(cfg["uav"]["num_uavs"])),
              "initial_batteries_j":bundle["initial_battery_j"],
              "malicious_ids":bundle["memberships"]["malicious"],"memberships":bundle["memberships"],
              "dataset":bundle["protocol"]["dataset"],"method":method,"policy":policy,
              "condition":bundle["protocol"]["condition"],"scenario_seed":bundle["protocol"]["scenario_seed"],
              "scenario_pairing_identity":bundle["rng"]["namespace"],"device":device,
              "purpose":bundle["protocol"]["purpose"],"command":sys.argv,
              "source_root":str(_source_root()),"original_root":str(Path(original_root).resolve()),
              "scenario_path":str(Path(scenario).resolve()),"data_content_id":bundle["dataset_manifest"]["content_id"],
              "budgets":{"host_memory_bytes":host_limit_bytes,"host_physical_memory_bytes":physical_ram,"vram_fraction":0.8,
                         "step_budget_start":starting_budget,"attempt_maximum_steps":bound},
              "observer_scope":"oracle_simulation","not_historical_reproduction":True}
    metadata.update(v5_metadata)
    writer=AttemptWriter(path,metadata)
    server=context=None
    precision_evidence={}
    started=time.perf_counter()
    try:
        try:
            with precision_policy_scope(bundle["precision_execution_identity"], evidence=precision_evidence):
                _device_gate(device)
                write_json_once(path/"import-origins.json",imports)
                execution_environment=_execution_environment(device)
                execution_environment["precision_execution_identity"]=bundle["precision_execution_identity"]
                execution_environment["precision_at_entry"]=copy.deepcopy(precision_evidence)
                execution_environment["precision_restoration_evidence"]="precision-execution.json"
                write_json_once(path/"environment.json",execution_environment)
                write_json_once(path/"scenario-reference.json",{"path":str(Path(scenario).resolve()),
                                "bundle_id":bundle["bundle_id"],"payloads":bundle["payloads"],"rng":bundle["rng"],
                                "data_content_id":bundle["dataset_manifest"]["content_id"],
                                "model_tensor_hash":bundle["model_tensor_hash"],
                                "model_execution_identity":bundle["model_execution_identity"],
                                "precision_execution_identity":bundle["precision_execution_identity"],
                                "indices_logical_manifest":bundle["indices_logical_manifest"],
                                "traces_logical_manifest":bundle["traces_logical_manifest"]})
                write_json_once(path/"scientific-memberships.json",bundle["memberships"])
                write_json_once(path/"initial-batteries.json",bundle["initial_battery_j"])
                server,context=build_components(bundle,scenario,method,policy,device,writer,step_budget,host_limit_bytes)
                if fullrun_adapter is not None:fullrun_adapter.install(server,context)
                write_json_once(path/"resolved-config.json",server.cfg.to_dict())
                context.memory=MemorySampler(device,host_limit_bytes);context.memory.start();context.memory.check_budget()
                with context.timer.region("initial_evaluation",sync=True):
                    context.query_purpose="initial_root"
                    initial_root=server._eval(server.val_loader)
                    initial_test=server._eval(server.test_loader)
                initial_memory=context.memory.stop();context.memory=None
                write_json_once(path/"initial-evaluation.json",{"boundary":"before_round0","round":-1,
                                "root_accuracy":initial_root["acc"],"root_loss":initial_root["loss"],
                                "test_accuracy":initial_test["acc"],"test_loss":initial_test["loss"],
                                "counters":context.counters,"timers":context.timer.values,"memory":initial_memory,
                                "model_tensor_hash":tensor_state_hash(server.model.state_dict())})
                if tensor_state_hash(server.model.state_dict())!=bundle["model_tensor_hash"]:
                    raise ValueError("Initial evaluation mutated the saved initial model")
                for round_id in range(cfg["train"]["rounds"]):
                    if fullrun_adapter is not None:fullrun_adapter.before_round(round_id)
                    server.run_round(round_id)
                    if fullrun_adapter is not None:fullrun_adapter.after_round(round_id)
                final=dict(context.final_record["evaluations"])
                if not all(key in final for key in ("test_accuracy","test_loss","root_accuracy","root_loss")):
                    raise ValueError("Required final-boundary test/root evaluation is missing")
                model_finite=all(bool(torch.isfinite(value).all()) for value in server.model.state_dict().values())
                if not model_finite:
                    from .context import ScientificFailure
                    raise ScientificFailure("numerical_model","final model is nonfinite",{"nonfinite_model_recorded":True})
                consumed=step_budget.completed-starting_budget["completed"]
                if consumed>bound["total"]:raise ValueError("Actual optimizer count exceeded frozen attempt bound")
                final_state={name:value.detach().cpu().clone() for name,value in server.model.state_dict().items()}
                with (path/"final-model.pt").open("xb") as stream:
                    torch.save(final_state,stream);stream.flush();os.fsync(stream.fileno())
                write_json_once(path/"final-model-identity.json",{"tensor_hash":tensor_state_hash(final_state),
                                "file_sha256":hashlib.sha256((path/"final-model.pt").read_bytes()).hexdigest(),
                                "all_tensors_finite":True})
        finally:
            write_json_once(path/"precision-execution.json",precision_evidence)
        write_json_once(path/"execution-summary.json",{"wall_seconds":time.perf_counter()-started,
                        "step_budget_start":starting_budget,"step_budget_end":step_budget.snapshot(),
                        "attempt_completed_optimizer_steps":consumed,"attempt_maximum_steps":bound,
                        "claim_eligible":False,"host_energy_measured":False})
        return writer.finalize(final,final_model_finite=True)
    except BaseException as exc:
        failure_memory=None
        if context is not None and context.memory is not None:
            try:failure_memory=context.memory.stop()
            except Exception as memory_error:failure_memory={"stop_error":repr(memory_error)}
            context.memory=None
        kind,evidence=_classify_failure(exc)
        evidence.update(exception_type=type(exc).__name__,exception_message=str(exc),
                        device=device,host_limit_bytes=host_limit_bytes,step_budget=step_budget.snapshot())
        failure_context={"error":repr(exc),"classification":kind,"evidence":evidence,
                         "wall_seconds":time.perf_counter()-started,"memory":failure_memory,
                         "round":context.round_idx if context is not None else None,
                         "counters":context.counters if context is not None else {},
                         "timers":context.timer.values if context is not None else {},
                         "active_timers":list(context.timer.active) if context is not None else [],
                         "partial_round":context.record if context is not None else None}
        if not writer.terminal:
            write_json_once(path/"failure-context.json",_failure_audit_value(failure_context))
            writer.fail(kind,evidence,str(exc))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    prepare=actions.add_parser("prepare",help="initialize/seal one offline scenario,zero optimizer steps")
    prepare.add_argument("--protocol",required=True)
    prepare.add_argument("--data-root",required=True)
    prepare.add_argument("--output",required=True)
    prepare.add_argument("--original-root",required=True)
    prepare.add_argument("--protect-root",action="append",default=[])
    run=actions.add_parser("run",help="SEPARATE LATER TRAINING ACTION; exactly one attempt")
    run.add_argument("--scenario",required=True)
    run.add_argument("--method",choices=METHODS,required=True)
    run.add_argument("--policy",choices=POLICIES,required=True)
    run.add_argument("--device",choices=("cpu","cuda"),required=True)
    run.add_argument("--output",required=True)
    run.add_argument("--original-root",required=True)
    run.add_argument("--protect-root",action="append",default=[])
    run.add_argument("--max-optimizer-steps",type=int,required=True)
    run.add_argument("--host-memory-limit-gib",type=float,required=True)
    run.add_argument("--training-authorized",action="store_true",help="acknowledge separately obtained authorization; this task does not authorize training")
    verify=actions.add_parser("verify",help="read-only existing attempt certificate verification")
    verify.add_argument("--attempt",required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=="prepare":
            from ..comparison.provenance import safe_new_target
            from .scenario import prepare as prepare_scenario
            _protection(args.original_root,data_root=args.data_root,protected_roots=args.protect_root)
            safe_new_target(args.output,protected=_protection(args.original_root,data_root=args.data_root,protected_roots=args.protect_root))
            assert_runtime_imports()
            result=prepare_scenario(load_protocol(args.protocol),args.data_root,args.output)
            print(json.dumps({"status":"prepared","scenario_id":result["bundle_id"],"optimizer_steps":0}))
        elif args.action=="run":
            result=run_entry(args.scenario,args.method,args.policy,args.device,args.output,
                  step_budget=StepBudget(args.max_optimizer_steps),
                  host_limit_bytes=int(args.host_memory_limit_gib*1024**3),original_root=args.original_root,
                  protected_roots=args.protect_root,training_authorized=args.training_authorized)
            print(json.dumps(result,allow_nan=False))
        else:
            result=validate_attempt(args.attempt)
            print(json.dumps(result,allow_nan=False))
            return 0 if result["complete"] else 2
        return 0
    except KeyboardInterrupt as exc:
        print(str(exc),file=sys.stderr);return 130
    except Exception as exc:
        print(type(exc).__name__+": "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":
    raise SystemExit(main())
