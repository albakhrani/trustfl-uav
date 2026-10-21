"""Pure standard-library v5 subset rules; no model or dataset dependencies.

All IDs below are ultimate IDs in the declared split namespace. Full partitions
are inputs from the unchanged scientific v1 allocator, never reconstructed here.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import math

VERSION = "sampled_clients_reserved_root_v5"
SPECS = {
    "emnist_byclass": {"classes":62,"cap":512,"reserve_per_class":112,"root_per_class":10},
    "eurosat": {"classes":10,"cap":256,"reserve_per_class":21,"root_per_class":21},
}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")


def subset_recipe(dataset, seed, content_id, base_namespace, namespace):
    if dataset not in SPECS or type(seed) is not int or seed < 0:
        raise ValueError("Unsupported typed v5 dataset/seed")
    for digest in (content_id,base_namespace):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Canonical SHA256 identity required")
    if not isinstance(namespace,str) or not namespace:
        raise ValueError("Ultimate train namespace required")
    return {"version":VERSION,"dataset":dataset,"scenario_seed":seed,
            "dataset_identity":content_id,"base_partition_namespace":base_namespace,
            "ultimate_namespace":namespace,"client_cap":SPECS[dataset]["cap"],
            "active_root_per_class":SPECS[dataset]["root_per_class"],
            "domains":["v5_client_cap","v5_root_subset"],
            "master":"SHA256(UTF8(TrustFLCommonPolicy.master.v1:+decimal_seed))",
            "ordering":"HMAC-SHA256 priority then ultimate ID; saved IDs ascending"}


def recipe_identity(recipe):
    return hashlib.sha256(canonical(recipe)).hexdigest()


def _ids(values):
    values=list(values)
    if any(type(i) is not int or i < 0 for i in values) or len(set(values))!=len(values):
        raise ValueError("Ultimate IDs must be unique nonnegative typed integers")
    return values


def select(values, count, recipe, domain, owner):
    """Deterministic uniform hash priorities; no draw depends on input ordering."""
    values=_ids(values)
    if type(count) is not int or not 0<=count<=len(values) or domain not in recipe["domains"] or type(owner) is not int or owner<0:
        raise ValueError("Invalid subset count/domain/owner")
    master=hashlib.sha256(("TrustFLCommonPolicy.master.v1:"+str(recipe["scenario_seed"])).encode("utf-8")).digest()
    def priority(item):
        event={"version":VERSION,"base_partition_namespace":recipe["base_partition_namespace"],
               "dataset_identity":recipe["dataset_identity"],"namespace":recipe["ultimate_namespace"],
               "event":domain,"ultimate_sample_id":item,
               "client_id" if domain=="v5_client_cap" else "class_id":owner}
        return hmac.new(master,canonical(event),hashlib.sha256).digest(),item
    return sorted(sorted(values,key=priority)[:count])


def histogram(values, labels, classes):
    result=[0]*classes
    for index in values:
        label=labels[index]
        if type(label) is not int or not 0<=label<classes:
            raise ValueError("Invalid class label")
        result[label]+=1
    return result


def variance(hist):
    n=sum(hist); classes=len(hist)
    if n<=0:
        raise ValueError("Empty retained client")
    # Same population variance of normalized class histogram as run.build_components.
    return sum((value/n-1/classes)**2 for value in hist)/classes*classes**2


def derive(full_reserve, full_parts, labels, recipe):
    spec=SPECS[recipe["dataset"]]; classes=spec["classes"]
    reserve=_ids(full_reserve); parts=[_ids(part) for part in full_parts]
    if len(parts)!=30 or any(len(part)<10 for part in parts):
        raise ValueError("V5 requires30 original nonempty/min10 client partitions")
    combined=reserve+[i for part in parts for i in part]
    if len(set(combined))!=len(combined) or any(i>=len(labels) for i in combined):
        raise ValueError("Full reserve/client partitions overlap or exceed labels")
    if histogram(reserve,labels,classes)!=[spec["reserve_per_class"]]*classes:
        raise ValueError("Root reserve is not the fixed class-balanced full reserve")
    root=[]
    for c in range(classes):
        root.extend(select([i for i in reserve if labels[i]==c],spec["root_per_class"],recipe,"v5_root_subset",c))
    root.sort(); rootset=set(root)
    retained=[select(part,min(spec["cap"],len(part)),recipe,"v5_client_cap",uid) for uid,part in enumerate(parts)]
    excluded=[]
    for part,keep in zip(parts,retained):
        keep_set=set(keep); excluded.append(sorted(set(part)-keep_set))
    unused=sorted(set(reserve)-rootset)
    fullhist=[histogram(part,labels,classes) for part in parts]
    keephist=[histogram(part,labels,classes) for part in retained]
    excludedhist=[histogram(part,labels,classes) for part in excluded]
    metadata={"version":VERSION,"recipe":recipe,"subset_namespace":recipe_identity(recipe),
              "original_counts":[len(p) for p in parts],"retained_counts":[len(p) for p in retained],
              "excluded_counts":[len(p) for p in excluded],"sigma2":[variance(h) for h in keephist],
              "full_client_class_histograms":fullhist,"retained_client_class_histograms":keephist,
              "excluded_client_class_histograms":excludedhist,"root_class_histogram":histogram(root,labels,classes),
              "reserve_class_histogram":histogram(reserve,labels,classes),
              "excluded_reserve_class_histogram":histogram(unused,labels,classes),
              "full_client_global_histogram":[sum(h[c] for h in fullhist) for c in range(classes)],
              "retained_client_global_histogram":[sum(h[c] for h in keephist) for c in range(classes)],
              "excluded_client_global_histogram":[sum(h[c] for h in excludedhist) for c in range(classes)]}
    return {"root":root,"retained":retained,"excluded":excluded,"unused_reserve":unused,"metadata":metadata}


def validate_partition(train, test, reserve, full_parts, retained, excluded, root,
                       unused_reserve, labels, recipe, namespaces, saved_metadata):
    train=_ids(train); test=_ids(test)
    full=reserve+[item for part in full_parts for item in part]
    if len(full)!=len(train) or set(full)!=set(train):
        raise ValueError("Full reserve/partitions do not exactly cover outer train")
    if namespaces["train"]==namespaces["test"] and set(train)&set(test):
        raise ValueError("Namespace-qualified train/test leakage")
    expected=derive(reserve,full_parts,labels,recipe)
    actual={"root":root,"retained":retained,"excluded":excluded,"unused_reserve":unused_reserve,"metadata":saved_metadata}
    if canonical(expected)!=canonical(actual):
        raise ValueError("Saved deterministic retained/excluded/root/count/histogram metadata mismatch")
    if any(not math.isfinite(v) for v in saved_metadata["sigma2"]):
        raise ValueError("Nonfinite actual-client variance")
    return True
