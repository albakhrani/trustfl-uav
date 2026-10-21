"""New immutable real-data scenarios. Preparation performs zero optimizer steps."""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
from ..comparison.provenance import safe_new_target, source_identity, tensor_state_hash
from .datasets import inspect_offline, partition_indices, validate_indices, sha256_file
from .protocol import DATASET_SPECS, identity, native_config_dict, pairing_recipe, validate_protocol
from .policy import generate_priorities, validate_priorities
from .randomness import CompactEventRandomness, domain_registry
from .precision import precision_identity, require_precision_identity, precision_policy_scope


def _write_json(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")


def initial_physics(protocol,rng):
    cfg=protocol["scientific"]["uav"]; count=cfg["num_uavs"]
    hm=sorted(rng.numpy("memberships",operation="high_mobility").choice(count,int(round(cfg["high_mobility_fraction"]*count)),replace=False).tolist())
    potential=sorted(rng.numpy("memberships",operation="potential_malicious").choice(count,int(round(0.4*count)),replace=False).tolist())
    active=potential if protocol["condition"]=="adaptive40" else []
    initial=[]; means={}
    for uid in range(count):
        g=rng.numpy("physical_initialization",uid=uid,operation="physical_state")
        position=[float(g.uniform(0,cfg["area_m"])),float(g.uniform(0,cfg["area_m"])),float(g.uniform(cfg["alt_min_m"],cfg["alt_max_m"]))]
        frequency=float(g.uniform(cfg["f_min_hz"],cfg["f_max_hz"]))
        battery=float(cfg["battery_j"]*g.uniform(0.8,1.2))
        initial.append({"uid":uid,"pos":position,"vel":[0.0,0.0,0.0],"energy_j":battery,
                        "f_actual_hz":frequency,"f_reported_hz":frequency,"high_mobility":uid in hm,
                        "alive":True,"last_compute_time_s":0.0,"last_committed_time_s":0.0,
                        "last_comm_time_s":0.0,"last_delivered":True,"energy_history":[]})
        vmax=cfg["v_max"]*(cfg["high_mobility_speed_factor"] if uid in hm else 1.0)
        mean=rng.numpy("physical_initialization",uid=uid,operation="gauss_markov_mean").normal(0,vmax/2,size=3)
        mean[2]*=0.2
        means[str(uid)]=mean.tolist()
    groups={"potential_malicious":potential,"malicious":active,"high_mobility":hm,
            "potential_overlap":sorted(set(potential)&set(hm)),"overlap":sorted(set(active)&set(hm)),
            "honest_high_mobility":sorted(set(hm)-set(active)),"prospective":True}
    return initial,means,groups


def trace_arrays(protocol,rng):
    cfg=protocol["scientific"]; rounds=cfg["train"]["rounds"]; count=cfg["uav"]["num_uavs"]
    result={"priority":generate_priorities(rng,rounds,count),
            "mobility":np.empty((rounds,count,3),dtype=np.float64),
            "channel":np.empty((rounds,count),dtype=np.float64),
            "attack_coins":np.empty((rounds,count,2),dtype=np.float64)}
    for t in range(rounds):
        rng.begin_round(t)
        for uid in range(count):
            result["mobility"][t,uid]=rng.numpy("mobility_innovation",t,uid).normal(size=3)
            result["channel"][t,uid]=rng.numpy("channel_unit_exponential_fading",t,uid).exponential(1.0)
            for i,event in enumerate(("attack_participation","attack_delay_branch")):
                result["attack_coins"][t,uid,i]=rng.numpy(event,t,uid).random()
    return result


def validate_traces(arrays,protocol):
    cfg=protocol["scientific"]; t=cfg["train"]["rounds"]; n=cfg["uav"]["num_uavs"]
    if set(arrays)!={"priority","mobility","channel","attack_coins"}:
        raise ValueError("Unexpected/missing numeric trace arrays")
    validate_priorities(arrays["priority"],t,n)
    for key,shape in (("mobility",(t,n,3)),("channel",(t,n)),("attack_coins",(t,n,2))):
        value=arrays[key]
        if value.shape!=shape or value.dtype!=np.dtype("float64") or not np.isfinite(value).all():
            raise ValueError("Malformed numeric trace: "+key)
    if np.any(arrays["channel"]<0) or np.any(arrays["attack_coins"]<0) or np.any(arrays["attack_coins"]>=1):
        raise ValueError("Numeric trace outside distribution range")
    return True


def logical_array_manifest(arrays):
    """Content identity independent of NPZ container metadata or host byte order."""
    result = {}
    for name, value in sorted(arrays.items()):
        value = np.asarray(value)
        if value.dtype.kind not in "biuf" or value.dtype.hasobject:
            raise ValueError("Numeric array required for logical identity")
        canonical = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
        result[name] = {"dtype": canonical.dtype.str, "shape": list(canonical.shape),
                        "sha256": hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()}
    return result


def _tensor_manifest(state):
    return {name:{"shape":list(value.shape),"dtype":str(value.dtype),"sha256":sha256_tensor(value)}
            for name,value in sorted(state.items())}


def sha256_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare(protocol,data_root,output):
    """Hash inputs, partition, initialize one model and seal artifacts; no SGD."""
    protocol=validate_protocol(protocol)
    if protocol["purpose"]=="v5":
        raise ValueError("V5 requires separate prepare_data_v5 and authorized initialize_v5 stages")
    source_root=Path(__file__).resolve().parents[2]
    target=safe_new_target(output,protected=[source_root,Path(data_root)])
    # Explicit constructor/dependency gate; never install or download on failure.
    import torch
    import torchvision
    from .models import build_model, model_identity
    from ..comparison.provenance import environment
    cfg=native_config_dict(protocol,data_root)
    info=inspect_offline(protocol["dataset"],data_root)
    recipe=pairing_recipe(protocol,info["manifest"]["content_id"])
    namespace=identity(recipe)
    rng=CompactEventRandomness(protocol["scenario_seed"],namespace)
    indexes,indexmeta=partition_indices(info,protocol,rng)
    physical,means,groups=initial_physics(protocol,rng)
    traces=trace_arrays(protocol,rng)
    # A bounded preflight before reserving/writing: compact indexes/traces plus
    # one float32 model and source manifests; no giant potential event table.
    projected=sum(x.nbytes for x in indexes.values())+sum(x.nbytes for x in traces.values())+DATASET_SPECS[protocol["dataset"]]["parameters"]*4+64*1024*1024
    if projected>512*1024*1024:
        raise RuntimeError("Scenario preparation projection exceeds512MiBbound")
    if shutil.disk_usage(target.parent).free<3*projected:
        raise OSError("Insufficient free space for protected scenario preparation")
    precision_evidence={}
    with precision_policy_scope(precision_identity(), evidence=precision_evidence):
        with rng.scope("initialization"):
            model=build_model(SimpleNamespace(**cfg["model"]),cfg["data"]["dataset"])
        state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        if sum(p.numel() for p in model.parameters())!=DATASET_SPECS[protocol["dataset"]]["parameters"]:
            raise ValueError("Constructed model parameter count differs from reviewed architecture")
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("Nonfinite initial model")
        model_tensor_hash=tensor_state_hash(state)
        model_tensors=_tensor_manifest(state)
        model_parameter_count=sum(p.numel() for p in model.parameters())
        preparation_environment=environment()
    preparation_environment["precision_execution_identity"]=precision_identity()
    preparation_environment["precision_execution"]=precision_evidence
    source=source_identity(source_root)
    bundle={"schema":"scientific_scenario_v1","prospective":True,"historical_recovery":False,
            "protocol":protocol,"protocol_hash":identity(protocol),"resolved_config":cfg,
            "rng":{"master_seed":protocol["scenario_seed"],"namespace":namespace},
            "pairing_recipe":recipe,"dataset_manifest":info["manifest"],"indices":indexmeta,
            "indices_logical_manifest":logical_array_manifest(indexes),
            "traces_logical_manifest":logical_array_manifest(traces),
            "memberships":groups,"initial_uav":physical,"gm_mean":means,
            "model_tensor_hash":model_tensor_hash,"model_tensors":model_tensors,
            "model_execution_identity":model_identity(SimpleNamespace(**cfg["model"]),cfg["data"]["dataset"]),
            "precision_execution_identity":precision_identity(),
            "model_parameter_count":model_parameter_count,
            "rng_registry":domain_registry(),"preparation_projected_bytes":projected,
            "initial_battery_j":{str(u["uid"]):u["energy_j"] for u in physical},
            "permutation_storage":"Unused events derivable; consumed int64 permutations written as attempt-local immutable chunks",
            "numpy_version":np.__version__,"torchvision_version":torchvision.__version__}
    target.mkdir()
    _write_json(target/"preparation_started.json",{"status":"preparing","prospective":True})
    try:
        with (target/"indexes.npz").open("xb") as stream:
            np.savez_compressed(stream,**indexes)
        with (target/"traces.npz").open("xb") as stream:
            np.savez_compressed(stream,**traces)
        with (target/"initial_model.pt").open("xb") as stream:
            torch.save(state,stream)
        _write_json(target/"source_identity.json",source)
        _write_json(target/"environment.json",preparation_environment)
        bundle["payloads"]={path.name:sha256_file(path) for path in sorted(target.iterdir())}
        bundle["bundle_id"]=identity(bundle)
        _write_json(target/"scenario.json",bundle)
        _write_json(target/"preparation_status.json",{"status":"complete","bundle_id":bundle["bundle_id"]})
    except BaseException as exc:
        _write_json(target/"preparation_failure.json",{"status":"failed","error":repr(exc)})
        raise
    return load_bundle(target,check_source=True)


def load_arrays(directory):
    directory=Path(directory)
    result=[]
    for name in ("indexes.npz","traces.npz"):
        with np.load(directory/name,allow_pickle=False) as stored:
            result.append({key:stored[key] for key in stored.files})
    return tuple(result)


def load_bundle(directory,check_source=False):
    directory=Path(directory)
    if not directory.is_absolute() or directory.is_symlink():
        raise ValueError("Absolute immutable scenario directory required")
    bundle=json.loads((directory/"scenario.json").read_text(encoding="utf-8"))
    is_v5=bundle.get("schema")=="scientific_scenario_v5" or bundle.get("protocol",{}).get("purpose")=="v5"
    if is_v5:
        bundle=_v5_read(directory/"scenario.json")
        if any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in (directory,*directory.parents)):
            raise ValueError("V5 scenario aliases are forbidden")
    require_precision_identity(bundle.get("precision_execution_identity"))
    saved=bundle.pop("bundle_id")
    if identity(bundle)!=saved:
        raise ValueError("Scenario manifest identity mismatch")
    bundle["bundle_id"]=saved
    expected_schema="scientific_scenario_v5" if bundle["protocol"].get("purpose")=="v5" else "scientific_scenario_v1"
    if bundle.get("schema")!=expected_schema or bundle.get("protocol_hash")!=identity(validate_protocol(bundle["protocol"])):
        raise ValueError("Scenario protocol/schema mismatch")
    status=_v5_read(directory/"preparation_status.json") if is_v5 else json.loads((directory/"preparation_status.json").read_text())
    if status!={"status":"complete","bundle_id":saved}:
        raise ValueError("Incomplete scenario preparation")
    required={"preparation_started.json","indexes.npz","traces.npz","initial_model.pt","source_identity.json","environment.json"}
    if set(bundle["payloads"])!=required:
        raise ValueError("Required immutable scenario payload inventory mismatch")
    for name,digest in bundle["payloads"].items():
        path=directory/name
        if path.is_symlink() or sha256_file(path)!=digest:
            raise ValueError("Scenario payload hash mismatch: "+name)
    manifest=dict(bundle["dataset_manifest"]); content=manifest.pop("content_id")
    if identity(manifest)!=content:
        raise ValueError("Dataset manifest content identity mismatch")
    if bundle["pairing_recipe"]!=pairing_recipe(bundle["protocol"],content) or identity(bundle["pairing_recipe"])!=bundle["rng"]["namespace"]:
        raise ValueError("Pairing recipe/namespace mismatch")
    if identity(bundle["resolved_config"])!=identity(native_config_dict(bundle["protocol"],bundle["resolved_config"]["data"]["data_dir"])):
        raise ValueError("Resolved scientific configuration mismatch")
    indexes,traces=load_arrays(directory)
    validate_indices(indexes,bundle["indices"]); validate_traces(traces,bundle["protocol"])
    if bundle["protocol"]["purpose"]=="v5":
        selection=bundle["indices"]["selection"]
        if selection["recipe"]["base_partition_namespace"]!=bundle["rng"]["namespace"] or selection["recipe"]["scenario_seed"]!=bundle["protocol"]["scenario_seed"] or selection["recipe"]["dataset"]!=bundle["protocol"]["dataset"] or selection["subset_namespace"]!=bundle.get("subset_namespace") or sha256_file(directory/"initial_model.pt")!=bundle.get("shared_initial_model_raw_sha256"):
            raise ValueError("V5 subset/model sharing identity mismatch")
    if logical_array_manifest(indexes)!=bundle.get("indices_logical_manifest") or logical_array_manifest(traces)!=bundle.get("traces_logical_manifest"):
        raise ValueError("Saved logical index/trace identities differ from actual numeric arrays")
    rng=CompactEventRandomness(bundle["rng"]["master_seed"],bundle["rng"]["namespace"])
    initial,means,groups=initial_physics(bundle["protocol"],rng)
    if initial!=bundle["initial_uav"] or means!=bundle["gm_mean"] or groups!=bundle["memberships"]:
        raise ValueError("Initial state/membership derivation mismatch")
    regenerated=trace_arrays(bundle["protocol"],rng)
    if any(not np.array_equal(traces[key],regenerated[key]) for key in traces):
        raise ValueError("Saved external trace differs from fixed event derivation")
    if check_source:
        source=_v5_read(directory/"source_identity.json") if is_v5 else json.loads((directory/"source_identity.json").read_text())
        if source["files"]!=source_identity(Path(__file__).resolve().parents[2])["files"]:
            raise ValueError("Scenario-producing source differs from executed source")
    return bundle


def preflight_tensors(bundle,directory,check_content=True):
    """Decode/check one initial state and offline views without any optimizer."""
    require_precision_identity(bundle.get("precision_execution_identity"))
    with precision_policy_scope(bundle["precision_execution_identity"]):
        import torch
        from .models import build_model, model_identity
        from .datasets import load_views
        expected_model_identity=model_identity(SimpleNamespace(**bundle["resolved_config"]["model"]),bundle["resolved_config"]["data"]["dataset"])
        if bundle.get("model_execution_identity")!=expected_model_identity:
            raise ValueError("Scenario model/operator identity differs from current scientific factory; prepare a fresh scenario")
        if check_content:
            observed=inspect_offline(bundle["protocol"]["dataset"],bundle["resolved_config"]["data"]["data_dir"])
            if observed["manifest"]!=bundle["dataset_manifest"]:
                raise ValueError("Current offline dataset content differs from immutable bundle")
        rng=CompactEventRandomness(bundle["rng"]["master_seed"],bundle["rng"]["namespace"])
        with rng.scope("initialization",operation="validation"):
            model=build_model(SimpleNamespace(**bundle["resolved_config"]["model"]),bundle["resolved_config"]["data"]["dataset"])
        state=torch.load(Path(directory)/"initial_model.pt",map_location="cpu",weights_only=True)
        if _tensor_manifest(state)!=bundle["model_tensors"] or tensor_state_hash(state)!=bundle["model_tensor_hash"]:
            raise ValueError("Initial-model logical tensor identity mismatch")
        reference=model.state_dict()
        if set(reference)!=set(state) or any(reference[key].shape!=value.shape or reference[key].dtype!=value.dtype or not bool(torch.isfinite(value).all()) for key,value in state.items()):
            raise ValueError("Initial-model structural/finite validation failed")
        model.load_state_dict(state,strict=True)
        return model,load_views(bundle,directory)


instantiate = preflight_tensors


# V5 keeps data preparation explicitly separate from any Torch/model creation.
def _v5_json(path, value):
    import os
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def _v5_read(path):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:
                raise ValueError("Duplicate manifest key: "+key)
            value[key]=item
        return value
    def bad(value):
        raise ValueError("Nonfinite manifest constant: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"),object_pairs_hook=pairs,parse_constant=bad)


def _v5_array_write(path, arrays):
    import os
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream,**arrays)
        stream.flush(); os.fsync(stream.fileno())


def _v5_copy_new(source, destination):
    """Publish a fresh copy and flush its writable handle (Windows compatible)."""
    import os
    with Path(source).open("rb") as original, Path(destination).open("xb") as copied:
        shutil.copyfileobj(original,copied)
        copied.flush(); os.fsync(copied.fileno())


def prepare_data_v5(protocol, data_root, output):
    """Future data-only stage: NumPy/hashing/traces; no Torch import or model.

    This API itself grants no execution permission. The staged controller must
    authenticate source, dataset root and storage allowances before invoking it.
    """
    protocol=validate_protocol(protocol)
    if protocol["purpose"]!="v5" or protocol["condition"]!="honest":
        raise ValueError("Shared v5 data preparation requires honest v5 protocol")
    source_root=Path(__file__).resolve().parents[2]
    target=safe_new_target(output,protected=[source_root,Path(data_root)])
    cfg=native_config_dict(protocol,data_root)
    info=inspect_offline(protocol["dataset"],data_root)
    recipe=pairing_recipe(protocol,info["manifest"]["content_id"])
    namespace=identity(recipe); rng=CompactEventRandomness(protocol["scenario_seed"],namespace)
    indexes,indexmeta=partition_indices(info,protocol,rng)
    physical,means,groups=initial_physics(protocol,rng)
    traces=trace_arrays(protocol,rng)
    projected=sum(x.nbytes for x in indexes.values())+sum(x.nbytes for x in traces.values())+64*1024*1024
    if projected>512*1024*1024 or shutil.disk_usage(target.parent).free<3*projected:
        raise OSError("Insufficient protected space for bounded v5 data preparation")
    source=source_identity(source_root)
    bundle={"schema":"scientific_data_scenario_v5","prospective":True,"historical_recovery":False,
            "numerical_initialization":"not_performed","protocol":protocol,"protocol_hash":identity(protocol),
            "resolved_config":cfg,"rng":{"master_seed":protocol["scenario_seed"],"namespace":namespace},
            "pairing_recipe":recipe,"subset_namespace":indexmeta["selection"]["subset_namespace"],
            "dataset_manifest":info["manifest"],"indices":indexmeta,
            "indices_logical_manifest":logical_array_manifest(indexes),"traces_logical_manifest":logical_array_manifest(traces),
            "memberships":groups,"initial_uav":physical,"gm_mean":means,
            "precision_execution_identity":precision_identity(),"rng_registry":domain_registry(),
            "preparation_projected_bytes":projected,"initial_battery_j":{str(u["uid"]):u["energy_j"] for u in physical},
            "permutation_storage":"Unused events derivable; consumed int64 permutations written as attempt-local immutable chunks",
            "numpy_version":np.__version__}
    target.mkdir()
    _v5_json(target/"preparation_started.json",{"status":"preparing_data_only","prospective":True})
    try:
        _v5_array_write(target/"indexes.npz",indexes); _v5_array_write(target/"traces.npz",traces)
        _v5_json(target/"source_identity.json",source)
        bundle["payloads"]={path.name:sha256_file(path) for path in sorted(target.iterdir())}
        bundle["bundle_id"]=identity(bundle)
        _v5_json(target/"data_scenario.json",bundle)
        _v5_json(target/"preparation_status.json",{"status":"data_complete","bundle_id":bundle["bundle_id"]})
    except BaseException as exc:
        _v5_json(target/"preparation_failure.json",{"status":"failed","error":repr(exc)})
        raise
    return load_data_v5(target,check_source=True)


def load_data_v5(directory, *, check_source=False, check_content=False):
    directory=Path(directory)
    if not directory.is_absolute() or any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in (directory,*directory.parents)):
        raise ValueError("Ordinary absolute v5 data scenario required")
    bundle=_v5_read(directory/"data_scenario.json"); saved=bundle.pop("bundle_id")
    if identity(bundle)!=saved:
        raise ValueError("V5 data bundle identity mismatch")
    bundle["bundle_id"]=saved
    p=validate_protocol(bundle["protocol"])
    if bundle["schema"]!="scientific_data_scenario_v5" or p["purpose"]!="v5" or p["condition"]!="honest" or bundle["protocol_hash"]!=identity(p) or bundle["numerical_initialization"]!="not_performed":
        raise ValueError("Invalid v5 data-only schema/protocol")
    require_precision_identity(bundle["precision_execution_identity"])
    if _v5_read(directory/"preparation_status.json")!={"status":"data_complete","bundle_id":saved}:
        raise ValueError("Incomplete v5 data preparation")
    required={"preparation_started.json","indexes.npz","traces.npz","source_identity.json"}
    if set(bundle["payloads"])!=required or set(path.name for path in directory.iterdir())!=required|{"data_scenario.json","preparation_status.json"}:
        raise ValueError("Unexpected v5 data-only payload inventory")
    for name,digest in bundle["payloads"].items():
        path=directory/name
        if path.is_symlink() or sha256_file(path)!=digest:
            raise ValueError("V5 data payload mismatch: "+name)
    content_manifest=dict(bundle["dataset_manifest"]); content=content_manifest.pop("content_id")
    if identity(content_manifest)!=content or bundle["pairing_recipe"]!=pairing_recipe(p,content) or bundle["rng"]!={"master_seed":p["scenario_seed"],"namespace":identity(bundle["pairing_recipe"])}:
        raise ValueError("V5 base namespace/content identity mismatch")
    if identity(bundle["resolved_config"])!=identity(native_config_dict(p,bundle["resolved_config"]["data"]["data_dir"])):
        raise ValueError("V5 resolved configuration mismatch")
    indexes,traces=load_arrays(directory)
    validate_indices(indexes,bundle["indices"]); validate_traces(traces,p)
    selection=bundle["indices"]["selection"]
    if selection["recipe"]["base_partition_namespace"]!=bundle["rng"]["namespace"] or selection["recipe"]["scenario_seed"]!=p["scenario_seed"] or selection["recipe"]["dataset"]!=p["dataset"] or bundle["subset_namespace"]!=selection["subset_namespace"]:
        raise ValueError("V5 subset/base namespace mismatch")
    if logical_array_manifest(indexes)!=bundle["indices_logical_manifest"] or logical_array_manifest(traces)!=bundle["traces_logical_manifest"]:
        raise ValueError("V5 logical array identity mismatch")
    rng=CompactEventRandomness(p["scenario_seed"],bundle["rng"]["namespace"])
    physical,means,groups=initial_physics(p,rng)
    if (physical,means,groups)!=(bundle["initial_uav"],bundle["gm_mean"],bundle["memberships"]):
        raise ValueError("V5 shared physical/membership mismatch")
    expected_traces=trace_arrays(p,rng)
    if any(not np.array_equal(traces[k],expected_traces[k]) for k in traces):
        raise ValueError("V5 trace derivation mismatch")
    if check_source and _v5_read(directory/"source_identity.json")["files"]!=source_identity(Path(__file__).resolve().parents[2])["files"]:
        raise ValueError("V5 data-producing source differs from candidate")
    if check_content and inspect_offline(p["dataset"],bundle["resolved_config"]["data"]["data_dir"])["manifest"]!=bundle["dataset_manifest"]:
        raise ValueError("V5 offline content changed")
    return bundle


def initialize_v5(data_directory, output, *, numerical_authorized=False):
    """Separately gated FUTURE initialization: one model, two condition bundles.

    Creates no optimizer and performs no model forward/backward. Data-only
    stages must never call this entry. A failed/colliding target is never reused.
    Returned mapping is {honest: path, adaptive40: path}; all methods reuse it.
    """
    if numerical_authorized is not True:
        raise PermissionError("V5 model initialization requires separate numerical authorization")
    data=load_data_v5(data_directory,check_source=True,check_content=True)
    source_root=Path(__file__).resolve().parents[2]
    target=safe_new_target(output,protected=[source_root,Path(data_directory),Path(data["resolved_config"]["data"]["data_dir"])])
    projected=2*(DATASET_SPECS[data["protocol"]["dataset"]]["parameters"]*4+
                 sum((Path(data_directory)/name).stat().st_size for name in ("indexes.npz","traces.npz")))+64*1024*1024
    if projected>512*1024*1024 or shutil.disk_usage(target.parent).free<3*projected:
        raise OSError("Insufficient protected v5 initialization space")
    target.mkdir()
    _v5_json(target/"initialization_started.json",{"status":"initializing","data_bundle_id":data["bundle_id"],"one_model_initialization":True})
    try:
        import os
        import copy
        import torch
        import torchvision
        from .models import build_model,model_identity
        from .protocol import make_protocol
        from ..comparison.provenance import environment
        cfg=data["resolved_config"]; rng=CompactEventRandomness(data["rng"]["master_seed"],data["rng"]["namespace"])
        precision_evidence={}
        with precision_policy_scope(precision_identity(),evidence=precision_evidence):
            with rng.scope("initialization"):
                model=build_model(SimpleNamespace(**cfg["model"]),cfg["data"]["dataset"])
            state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            parameter_count=sum(p.numel() for p in model.parameters())
            if parameter_count!=DATASET_SPECS[data["protocol"]["dataset"]]["parameters"] or any(not bool(torch.isfinite(v).all()) for v in state.values()):
                raise ValueError("V5 model initialization count/finite mismatch")
            tensor_hash=tensor_state_hash(state); tensors=_tensor_manifest(state)
            execution_identity=model_identity(SimpleNamespace(**cfg["model"]),cfg["data"]["dataset"])
            env=environment()
        env["precision_execution_identity"]=precision_identity(); env["precision_execution"]=precision_evidence
        (target/"honest").mkdir()
        model_file=target/"honest"/"initial_model.pt"
        with model_file.open("xb") as stream:
            torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
        model_raw_hash=sha256_file(model_file); del state,model
        outputs={}; bundle_ids={}
        for condition in ("honest","adaptive40"):
            child=target/condition
            if condition!="honest":
                child.mkdir()
            protocol=make_protocol(data["protocol"]["dataset"],data["protocol"]["scenario_seed"],condition,"v5","cuda")
            bundle=copy.deepcopy(data)
            for key in ("bundle_id","payloads","numerical_initialization"):
                del bundle[key]
            bundle.update(schema="scientific_scenario_v5",protocol=protocol,protocol_hash=identity(protocol),
                          resolved_config=native_config_dict(protocol,cfg["data"]["data_dir"]),
                          data_preparation_id=data["bundle_id"],model_tensor_hash=tensor_hash,model_tensors=tensors,
                          model_execution_identity=execution_identity,model_parameter_count=parameter_count,
                          torchvision_version=torchvision.__version__,shared_initial_model_raw_sha256=model_raw_hash)
            physical,means,groups=initial_physics(protocol,CompactEventRandomness(protocol["scenario_seed"],bundle["rng"]["namespace"]))
            bundle.update(initial_uav=physical,gm_mean=means,memberships=groups)
            _v5_json(child/"preparation_started.json",{"status":"preparing","prospective":True,"data_bundle_id":data["bundle_id"]})
            for name in ("indexes.npz","traces.npz","source_identity.json"):
                _v5_copy_new(Path(data_directory)/name,child/name)
            if condition!="honest":
                _v5_copy_new(model_file,child/"initial_model.pt")
            # The content is copied, never initialized again, for independent immutable bundles.
            _v5_json(child/"environment.json",env)
            bundle["payloads"]={path.name:sha256_file(path) for path in sorted(child.iterdir())}
            bundle["bundle_id"]=identity(bundle)
            _v5_json(child/"scenario.json",bundle)
            _v5_json(child/"preparation_status.json",{"status":"complete","bundle_id":bundle["bundle_id"]})
            load_bundle(child,check_source=True)
            outputs[condition]=str(child); bundle_ids[condition]=bundle["bundle_id"]
        _v5_json(target/"initialization_complete.json",{"schema":"scientific_initialization_v5","status":"complete",
                  "data_bundle_id":data["bundle_id"],"one_model_initialization":True,"model_tensor_hash":tensor_hash,
                  "model_raw_sha256":model_raw_hash,"condition_bundle_ids":bundle_ids})
        return outputs
    except BaseException as exc:
        _v5_json(target/"initialization_failure.json",{"status":"failed","error":repr(exc)})
        raise
