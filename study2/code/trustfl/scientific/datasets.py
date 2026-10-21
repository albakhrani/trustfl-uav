"""Offline content identities, ultimate index arrays and deterministic views.

Raw inventory/partition operations require NumPy only. Torch/torchvision are
imported only for real model/data construction. There is no download path.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import numpy as np
from .protocol import DATASET_SPECS, V5_PROFILE, identity, pairing_recipe, validate_protocol

TRANSFORM_CONTRACT = {
    "version":"preserved_deterministic_v1", "stochastic_operations":[],
    "mnist":{"ToTensor":True,"mean":[0.1307],"std":[0.3081]},
    "emnist_byclass":{"ToTensor":True,"mean":[0.1307],"std":[0.3081]},
    "eurosat":{"ToTensor":True,"mean":[0.3441,0.3801,0.4075],"std":[0.2028,0.1372,0.1157]},
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _input_file(path, root):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in (path,*path.parents)):
        raise FileNotFoundError("Missing/offline or symlinked dataset input: "+str(path))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes explicit root")
    return {"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)}


def _idx(image, label, count, classes):
    with image.open("rb") as f:
        header=f.read(16)
    if len(header)!=16 or struct.unpack(">4I",header)!=(2051,count,28,28) or image.stat().st_size!=16+count*784:
        raise ValueError("IDX image header/count/length mismatch: "+str(image))
    with label.open("rb") as f:
        header=f.read(8)
        values=np.frombuffer(f.read(),dtype=np.uint8).astype(np.int64)
    if len(header)!=8 or struct.unpack(">2I",header)!=(2049,count) or len(values)!=count:
        raise ValueError("IDX label header/count/length mismatch: "+str(label))
    if np.any(values>=classes) or set(values.tolist())!=set(range(classes)):
        raise ValueError("Unexpected IDX class labels")
    return values


def eurosat_split_indices(labels):
    labels=np.asarray(labels,dtype=np.int64)
    rng=np.random.default_rng(42)
    train,test=[],[]
    for c in np.unique(labels):
        values=np.where(labels==c)[0]
        rng.shuffle(values)
        split=int(round(len(values)*0.2))
        test.extend(values[:split].tolist())
        train.extend(values[split:].tolist())
    return np.asarray(sorted(train),dtype=np.int64),np.asarray(sorted(test),dtype=np.int64)


def inspect_offline(dataset, data_root):
    """Hash every consumed source file and validate raw labels/header structure.

    Hashing authenticates local bytes, not an upstream publisher signature or
    image-decoder/pixel-integrity validation. No writes and no scientific imports.
    """
    if dataset not in DATASET_SPECS:
        raise ValueError("Unknown offline scientific dataset")
    root=Path(data_root)
    if not root.is_absolute() or not root.is_dir() or any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in (root,*root.parents)):
        raise ValueError("Existing absolute offline root required")
    root=root.resolve()
    spec=DATASET_SPECS[dataset]
    files=[]
    if dataset in ("mnist","emnist_byclass"):
        folder=root/("MNIST" if dataset=="mnist" else "EMNIST")/"raw"
        prefix="" if dataset=="mnist" else "emnist-byclass-"
        names=(prefix+"train-images-idx3-ubyte",prefix+"train-labels-idx1-ubyte",
               prefix+("t10k" if dataset=="mnist" else "test")+"-images-idx3-ubyte",
               prefix+("t10k" if dataset=="mnist" else "test")+"-labels-idx1-ubyte")
        paths=[folder/name for name in names]
        files=[_input_file(path,root) for path in paths]
        train_labels=_idx(paths[0],paths[1],spec["train_count"],spec["classes"])
        test_labels=_idx(paths[2],paths[3],spec["test_count"],spec["classes"])
        train_ids=np.arange(len(train_labels),dtype=np.int64)
        test_ids=np.arange(len(test_labels),dtype=np.int64)
        namespaces={"train":dataset+"/train","test":dataset+"/test"}
        classes={str(i):i for i in range(spec["classes"])}
    else:
        folder=root/"eurosat"/"2750"
        if not folder.is_dir():
            raise FileNotFoundError("EuroSAT images required offline at "+str(folder))
        class_dirs=sorted(p for p in folder.iterdir() if p.is_dir())
        if len(class_dirs)!=10 or any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in class_dirs):
            raise ValueError("EuroSAT must contain ten ordinary class directories")
        classes={p.name:i for i,p in enumerate(class_dirs)}
        train_labels=[]
        for class_dir in class_dirs:
            images=sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg",".jpeg"))
            for path in images:
                files.append(_input_file(path,root))
                train_labels.append(classes[class_dir.name])
        if len(files)!=27000:
            raise ValueError("EuroSAT requires exactly27000offline JPEG files")
        train_labels=np.asarray(train_labels,dtype=np.int64)
        test_labels=train_labels.copy()
        train_ids,test_ids=eurosat_split_indices(train_labels)
        if (len(train_ids),len(test_ids))!=(21600,5400):
            raise ValueError("Unexpected stratified EuroSAT split counts")
        namespaces={"train":"eurosat/full","test":"eurosat/full"}
    manifest={"schema":"scientific_offline_content_v1","dataset":dataset,"files":files,
              "class_to_idx":classes,"namespaces":namespaces,
              "train_count":len(train_ids),"test_count":len(test_ids),
              "transforms":TRANSFORM_CONTRACT[dataset],"transform_version":TRANSFORM_CONTRACT["version"],
              "upstream_authentication":"not established by this local content hash"}
    manifest["content_id"]=identity(manifest)
    return {"manifest":manifest,"data_root":str(root),"train_ids":train_ids,"test_ids":test_ids,
            "train_labels":train_labels,"test_labels":test_labels}


def _class_balanced_root(train_ids, labels, seed):
    labels=np.asarray(labels)
    classes=np.unique(labels[train_ids])
    per_class=max(max(int(len(train_ids)*0.01),50)//len(classes),1)
    rng=np.random.default_rng(seed)
    root=[]
    for c in classes:
        candidates=train_ids[labels[train_ids]==c]
        root.extend(rng.choice(candidates,size=min(per_class,len(candidates)),replace=False).tolist())
    return np.asarray(sorted(root),dtype=np.int64)


def partition_indices(info, protocol, rng):
    """Class-balanced root first, then original bounded Dirichlet mathematics.

    Unlike the ambiguous legacy nested Subset path, every returned index is
    already an ultimate train/full-collection ID, never a view-local index.
    """
    p=validate_protocol(protocol)
    train=info["train_ids"].copy(); test=info["test_ids"].copy()
    labels=info["train_labels"]
    n=p["scientific"]["uav"]["num_uavs"]
    root=_class_balanced_root(train,labels,rng.torch_seed("root_and_client_partition",operation="root"))
    remaining=np.setdiff1d(train,root)
    if p["purpose"]=="v6":
        # Micro fixture intentionally is not the full class-balanced root rule.
        # Retain sixteen deterministic root IDs from the full-rule root; sixteen
        # examples/client from the disjoint remainder, and sixteen test IDs.
        root=np.asarray(rng.numpy("root_and_client_partition",operation="v6_root").choice(root,16,replace=False),dtype=np.int64)
        chosen=rng.numpy("root_and_client_partition",operation="v6_client").choice(remaining,n*16,replace=False)
        parts=[np.sort(v.astype(np.int64)) for v in np.array_split(chosen,n)]
        test=np.sort(rng.numpy("root_and_client_partition",operation="v6_test").choice(test,16,replace=False).astype(np.int64))
        train=np.sort(np.concatenate([root,*parts])); root=np.sort(root)
        attempts=1
    else:
        expected=DATASET_SPECS[p["dataset"]]["root_count"]
        if len(root)!=expected:
            raise ValueError("Root class-balanced count differs from fixed scientific protocol")
        g=np.random.default_rng(rng.torch_seed("root_and_client_partition",operation="client"))
        classes=np.unique(labels[remaining])
        for attempts in range(1,101):
            parts=[[] for _ in range(n)]
            for c in classes:
                ids=remaining[labels[remaining]==c].copy()
                g.shuffle(ids)
                proportions=g.dirichlet(np.full(n,0.2))
                cuts=(np.cumsum(proportions)*len(ids)).astype(int)[:-1]
                for uid,chunk in enumerate(np.split(ids,cuts)):
                    parts[uid].extend(chunk.tolist())
            if min(map(len,parts))>=10:
                parts=[np.asarray(sorted(part),dtype=np.int64) for part in parts]
                break
        else:
            raise RuntimeError("Dirichlet failed min10within100attempts; no automatic parameter change")
    arrays={"train_ids":train,"test_ids":test,"root_ids":root,
            "client_values":np.concatenate(parts),"client_offsets":np.cumsum([0]+[len(v) for v in parts],dtype=np.int64),
            "train_labels":info["train_labels"].astype(np.int64),"test_labels":info["test_labels"].astype(np.int64)}
    metadata={"schema":"scientific_indices_v1","content_id":info["manifest"]["content_id"],
              "namespaces":info["manifest"]["namespaces"],"clients":n,
              "partition_attempts":attempts,"profile":"full" if p["purpose"]=="v5" else p["data_profile"],
              "counts":{"train":len(train),"test":len(test),"root":len(root),"clients":[len(v) for v in parts]}}
    validate_indices(arrays,metadata)
    if p["purpose"]=="v5":
        return _cap_v5(arrays,metadata,p)
    return arrays,metadata


def client_parts(arrays):
    values,offsets=arrays["client_values"],arrays["client_offsets"]
    return [values[int(a):int(b)] for a,b in zip(offsets[:-1],offsets[1:])]


def validate_indices(arrays,metadata):
    if metadata.get("schema")=="scientific_indices_v5":
        return _validate_v5(arrays,metadata)
    required={"train_ids","test_ids","root_ids","client_values","client_offsets","train_labels","test_labels"}
    if set(arrays)!=required:
        raise ValueError("Unexpected/missing index arrays")
    for name,value in arrays.items():
        if not isinstance(value,np.ndarray) or value.ndim!=1 or value.dtype!=np.dtype("int64") or np.any(value<0):
            raise ValueError("Index arrays must be nonnegative 1Dint64: "+name)
    offsets=arrays["client_offsets"]
    if len(offsets)!=metadata["clients"]+1 or offsets[0]!=0 or offsets[-1]!=len(arrays["client_values"]) or np.any(np.diff(offsets)<=0):
        raise ValueError("Invalid client-offset coverage")
    root=arrays["root_ids"]; clients=arrays["client_values"]; train=arrays["train_ids"]; test=arrays["test_ids"]
    for values,labels in ((train,arrays["train_labels"]),(test,arrays["test_labels"]),(root,arrays["train_labels"]),(clients,arrays["train_labels"])):
        if len(np.unique(values))!=len(values) or (len(values) and values.max()>=len(labels)):
            raise ValueError("Repeated/out-of-range ultimate index")
    if np.intersect1d(root,clients).size or not np.array_equal(np.sort(np.concatenate((root,clients))),np.sort(train)):
        raise ValueError("Root/client leakage or incomplete training coverage")
    if metadata["namespaces"]["train"]==metadata["namespaces"]["test"] and np.intersect1d(train,test).size:
        raise ValueError("Outer train/test leakage")
    counts={"train":len(train),"test":len(test),"root":len(root),"clients":[len(x) for x in client_parts(arrays)]}
    if metadata["counts"]!=counts:
        raise ValueError("Saved index count manifest mismatch")
    return True



def _flat_parts(parts):
    return (np.asarray([i for part in parts for i in part],dtype=np.int64),
            np.cumsum([0]+[len(part) for part in parts],dtype=np.int64))


def _cap_v5(base,base_meta,protocol):
    from .subsets_v5 import derive, subset_recipe
    content=base_meta["content_id"]
    recipe=subset_recipe(protocol["dataset"],protocol["scenario_seed"],content,
                         identity(pairing_recipe(protocol,content)),base_meta["namespaces"]["train"])
    parts=[part.tolist() for part in client_parts(base)]
    picked=derive(base["root_ids"].tolist(),parts,base["train_labels"].tolist(),recipe)
    arrays=dict(base)
    arrays.update(full_root_reserve=base["root_ids"].copy(),
                  full_client_values=base["client_values"].copy(),
                  full_client_offsets=base["client_offsets"].copy(),
                  root_ids=np.asarray(picked["root"],dtype=np.int64),
                  excluded_root_ids=np.asarray(picked["unused_reserve"],dtype=np.int64))
    arrays["client_values"],arrays["client_offsets"]=_flat_parts(picked["retained"])
    arrays["excluded_client_values"],arrays["excluded_client_offsets"]=_flat_parts(picked["excluded"])
    metadata={"schema":"scientific_indices_v5","content_id":content,
              "namespaces":base_meta["namespaces"],"clients":30,"profile":V5_PROFILE,
              "partition_attempts":base_meta["partition_attempts"],"base_metadata":base_meta,
              "selection":picked["metadata"],"counts":{"train":len(base["train_ids"]),
              "test":len(base["test_ids"]),"root":len(picked["root"]),
              "clients":picked["metadata"]["retained_counts"]}}
    validate_indices(arrays,metadata)
    return arrays,metadata


def _validate_v5(arrays,metadata):
    from .subsets_v5 import validate_partition, subset_recipe
    required={"train_ids","test_ids","root_ids","client_values","client_offsets","train_labels","test_labels",
              "full_root_reserve","full_client_values","full_client_offsets","excluded_root_ids",
              "excluded_client_values","excluded_client_offsets"}
    if set(arrays)!=required:
        raise ValueError("Unexpected/missing v5 index arrays")
    for name,value in arrays.items():
        if not isinstance(value,np.ndarray) or value.ndim!=1 or value.dtype!=np.dtype("int64") or np.any(value<0):
            raise ValueError("V5 arrays must be nonnegative1Dint64: "+name)
    base={k:arrays[k] for k in ("train_ids","test_ids","train_labels","test_labels")}
    base.update(root_ids=arrays["full_root_reserve"],client_values=arrays["full_client_values"],client_offsets=arrays["full_client_offsets"])
    if metadata["base_metadata"]["schema"]!="scientific_indices_v1" or metadata["base_metadata"]["profile"]!="full":
        raise ValueError("V5 base must be unchanged full scientific recipe")
    validate_indices(base,metadata["base_metadata"])
    selection=metadata["selection"]; recipe=selection["recipe"]; dataset=recipe["dataset"]
    if dataset not in ("emnist_byclass","eurosat"):
        raise ValueError("Invalid v5 dataset")
    spec=DATASET_SPECS[dataset]
    for key in ("train_labels","test_labels"):
        if set(arrays[key].tolist())!=set(range(spec["classes"])):
            raise ValueError("Missing or unexpected raw v5 dataset classes")
    if (len(arrays["train_ids"]),len(arrays["test_ids"]),metadata["clients"])!=(spec["train_count"],spec["test_count"],30):
        raise ValueError("Unexpected v5 full train/test/client counts")
    if recipe!=subset_recipe(dataset,recipe["scenario_seed"],metadata["content_id"],recipe["base_partition_namespace"],metadata["namespaces"]["train"]):
        raise ValueError("Invalid v5 subset namespace recipe")
    def unpack(prefix):
        values=arrays[prefix+"client_values"]; offsets=arrays[prefix+"client_offsets"]
        if len(offsets)!=31 or offsets[0]!=0 or offsets[-1]!=len(values) or np.any(np.diff(offsets)<0):
            raise ValueError("Invalid v5 client offsets")
        return [values[int(a):int(b)].tolist() for a,b in zip(offsets[:-1],offsets[1:])]
    validate_partition(arrays["train_ids"].tolist(),arrays["test_ids"].tolist(),arrays["full_root_reserve"].tolist(),
                       unpack("full_"),unpack(""),unpack("excluded_"),arrays["root_ids"].tolist(),
                       arrays["excluded_root_ids"].tolist(),arrays["train_labels"].tolist(),recipe,
                       metadata["namespaces"],selection)
    expected={"schema":"scientific_indices_v5","content_id":metadata["base_metadata"]["content_id"],
              "namespaces":metadata["base_metadata"]["namespaces"],"clients":30,"profile":V5_PROFILE,
              "partition_attempts":metadata["base_metadata"]["partition_attempts"],"base_metadata":metadata["base_metadata"],
              "selection":selection,"counts":{"train":len(arrays["train_ids"]),"test":len(arrays["test_ids"]),
              "root":len(arrays["root_ids"]),"clients":[len(p) for p in unpack("")]}}
    if identity(metadata)!=identity(expected):
        raise ValueError("V5 count/namespace metadata mismatch")
    return True


def base_identity(dataset,index,namespace=None):
    """Resolve arbitrary nested index views to the ultimate content/split/item."""
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<len(dataset):
        raise ValueError("Typed valid index required")
    i=int(index); seen=set()
    while hasattr(dataset,"dataset"):
        if id(dataset) in seen or not hasattr(dataset,"indices"):
            raise ValueError("Unsupported or cyclic dataset view")
        seen.add(id(dataset)); value=dataset.indices[i]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise ValueError("Nested mapping must contain integer indices")
        i=int(value); dataset=dataset.dataset
        if not 0<=i<len(dataset):
            raise ValueError("Nested index out of bounds")
    ns=getattr(dataset,"cp_namespace",namespace)
    content=getattr(dataset,"scientific_content_id",None)
    if not ns or (namespace is not None and ns!=namespace) or not content:
        raise ValueError("Missing/mismatched ultimate content or split identity")
    result={"content_id":content,"namespace":ns,"index":i}
    if hasattr(dataset,"scientific_relative_paths"):
        result["relative_path"]=dataset.scientific_relative_paths[i]
        result["file_sha256"]=dataset.scientific_file_hashes[i]
    return result


class IndexView:
    """Small explicit view usable with raw fixtures or Torch DataLoader."""
    def __init__(self,dataset,indices):
        values=np.asarray(indices)
        if values.ndim!=1 or values.dtype.kind not in "iu":
            raise ValueError("Explicit view indices must be a one-dimensional integer array")
        self.dataset=dataset; self.indices=values.astype(np.int64,copy=False)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        return self.dataset[int(self.indices[index])]


class RawIDXDataset:
    """Read authenticated raw IDX bytes directly, ignoring all processed caches.

    This uses the existing PIL->ToTensor->Normalize representation. A read-only
    memory map avoids constructing a second full EMNIST pixel copy in RAM.
    """
    def __init__(self, image_path, label_path, transform):
        from PIL import Image
        self.transform=transform
        with Path(image_path).open("rb") as stream:
            magic,count,rows,cols=struct.unpack(">4I",stream.read(16))
        if magic!=2051 or (rows,cols)!=(28,28):
            raise ValueError("Invalid raw IDX image header")
        with Path(label_path).open("rb") as stream:
            labelmagic,labelcount=struct.unpack(">2I",stream.read(8))
            self.targets=np.frombuffer(stream.read(),dtype=np.uint8).astype(np.int64)
        if labelmagic!=2049 or labelcount!=count or len(self.targets)!=count:
            raise ValueError("Raw IDX label count mismatch")
        self.images=np.memmap(image_path,mode="r",dtype=np.uint8,offset=16,shape=(count,rows,cols))
    def __len__(self):
        return len(self.targets)
    def __getitem__(self,index):
        from PIL import Image
        image=Image.fromarray(np.asarray(self.images[index]),mode="L")
        return self.transform(image),int(self.targets[index])


def load_offline_bases(dataset,data_root,manifest):
    """Construct existing deterministic transforms with downloads hard-disabled."""
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    root=Path(data_root)
    spec=TRANSFORM_CONTRACT[dataset]
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(tuple(spec["mean"]),tuple(spec["std"]))])
    if dataset=="mnist":
        files=manifest["files"]
        train=RawIDXDataset(root/files[0]["path"],root/files[1]["path"],transform)
        test=RawIDXDataset(root/files[2]["path"],root/files[3]["path"],transform)
    elif dataset=="emnist_byclass":
        files=manifest["files"]
        train=RawIDXDataset(root/files[0]["path"],root/files[1]["path"],transform)
        test=RawIDXDataset(root/files[2]["path"],root/files[3]["path"],transform)
    elif dataset=="eurosat":
        train=datasets.ImageFolder(str(root/"eurosat"/"2750"),transform=transform)
        test=train
        expected=[row["path"] for row in manifest["files"]]
        actual=[Path(path).relative_to(root).as_posix() for path,_ in train.samples]
        if actual!=expected or train.class_to_idx!=manifest["class_to_idx"]:
            raise ValueError("ImageFolder order/class mapping differs from content manifest")
        train.scientific_relative_paths=actual
        train.scientific_file_hashes=[row["sha256"] for row in manifest["files"]]
    else:
        raise ValueError("Unsupported dataset")
    for item,namespace in ((train,manifest["namespaces"]["train"]),(test,manifest["namespaces"]["test"])):
        item.cp_namespace=namespace
        item.scientific_content_id=manifest["content_id"]
    return train,test


def load_views(bundle,directory,context=None):
    """Return Client-compatible (base,ultimate-index) pairs plus root/test views.

    All accepted transforms are deterministic, so no sample-scoped PRF events
    are needed. The immutable input inventory and saved indices identify access.
    Existing Client's engineering EventDataset must be bypassed by its separate
    scientific subclass; do not silently send these through per-sample logging.
    """
    import torch
    from torch.utils.data import Subset
    from .scenario import load_arrays
    arrays,_=load_arrays(directory)
    validate_indices(arrays,bundle["indices"])
    train,test=load_offline_bases(bundle["protocol"]["dataset"],bundle["resolved_config"]["data"]["data_dir"],bundle["dataset_manifest"])
    train_labels=np.asarray(train.targets,dtype=np.int64)
    test_labels=np.asarray(test.targets,dtype=np.int64)
    if not np.array_equal(train_labels,arrays["train_labels"]) or not np.array_equal(test_labels,arrays["test_labels"]):
        raise ValueError("Constructed dataset labels differ from saved raw content")
    return ([(train,part.copy()) for part in client_parts(arrays)],
            Subset(train,arrays["root_ids"].tolist()),Subset(test,arrays["test_ids"].tolist()))
