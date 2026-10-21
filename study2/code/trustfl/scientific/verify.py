"""Fixed nontraining verification and separately invoked future V6. No auto grid."""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

COUNTS={'V1datasets':3,'V1lookups':72,'V2schedules':30,'V3cases':12,'V4cases':10,'V5cases':8,'stats':6}


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)


def source_snapshot(root):
    paths=sorted([*(root/'trustfl').rglob('*.py'),*(root/'tests'/'scientific').glob('*.py'),*(root/'scripts'/'scientific_transfer').glob('*')])
    return {str(p.relative_to(root)):{'sha256_raw':hashlib.sha256(p.read_bytes()).hexdigest(),
                'sha256_lf_normalized':hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()} for p in paths if p.is_file()}


def gate_source(expected):
    root=Path(__file__).resolve().parents[2]
    if root!=Path(expected).resolve() or Path.cwd().resolve()!=root or os.environ.get('PYTHONPATH'):
        raise RuntimeError('Explicit candidate cwd/import path required; PYTHONPATH overrides refused')
    return root


def module_from_file(root,name):
    path=root/'tests'/'scientific'/(name+'.py')
    spec=importlib.util.spec_from_file_location('_scientific_fixture_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@contextmanager
def zero_optimizer_guard():
    """Fail before any optimizer can be constructed; no optimizer exists in cases."""
    audit={'optimizer_steps':0,'optimizer_construction_attempts':0,'torch_present':False}
    if importlib.util.find_spec('torch') is None:
        audit['enforcement']='Torch absent: scientific optimizer execution unavailable'
        yield audit;return
    import torch
    audit['torch_present']=True
    old=torch.optim.Optimizer.__init__
    def forbidden(*args,**kwargs):
        audit['optimizer_construction_attempts']+=1
        raise RuntimeError('Nontraining suite prohibits optimizer construction and steps')
    torch.optim.Optimizer.__init__=forbidden
    audit['enforcement']='torch.optim.Optimizer.__init__ forbidden for entire fixed suite'
    try:yield audit
    finally:torch.optim.Optimizer.__init__=old


def run_unittests(module,cls_name,prefix,expected):
    cls=getattr(module,cls_name)
    names=unittest.defaultTestLoader.getTestCaseNames(cls)
    if len(names)!=expected or any(not n.startswith('test_'+prefix) for n in names):
        raise ValueError('Fixed fixture registry changed: '+cls_name)
    records=[]
    for name in names:
        result=unittest.TestResult();cls(name).run(result)
        status='FAIL' if result.errors or result.failures else ('BLOCKED' if result.skipped else 'PASS')
        records.append({'case_id':name[5:],'status':status,'optimizer_steps':0,
            'errors':[msg for _,msg in result.errors+result.failures],
            'skipped':[msg for _,msg in result.skipped]})
    return records


def nontraining(root,data_root,out):
    started=time.perf_counter();before=source_snapshot(root)
    save(out/'source-before.json',before)
    fixtures=[];details={};origins={}
    for path in sorted((root/'trustfl'/'scientific').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    for name in ('trustfl.scientific.protocol','trustfl.scientific.datasets','trustfl.scientific.scenario',
                 'trustfl.scientific.methods','trustfl.scientific.records','trustfl.scientific.analysis',
                 'trustfl.scientific.context','trustfl.scientific.instrumentation','trustfl.scientific.run'):
        mod=importlib.import_module(name);path=Path(mod.__file__).resolve()
        if not path.is_relative_to(root):raise RuntimeError('Foreign import '+name)
        origins[name]=str(path)
    with zero_optimizer_guard() as guard:
        m=module_from_file(root,'test_v1_v2')
        v1=m.run_v1(data_root,out/'V1');details['V1']=v1;fixtures.extend(v1['datasets'])
        v2=m.run_v2();details['V2']=v2;fixtures.extend(v2['cases'])
        tensor=module_from_file(root,'test_v3_v5')
        v3=tensor.run_v3();fixtures.extend(v3);details['V3']=v3
        v4=run_unittests(module_from_file(root,'test_v4'),'IntegrityFixtures','V4_',10)
        fixtures.extend(v4);details['V4']=v4
        v5=tensor.run_v5();fixtures.extend(v5);details['V5']=v5
        stats=run_unittests(module_from_file(root,'test_statistics'),'StatisticalFixtures','ST',6)
        fixtures.extend(stats);details['statistics']=stats
        # Separate V5 identities, not an extra algebra/statistical fixture.
        from .datasets import inspect_offline
        identities=[]
        for entry in v1['datasets']:
            dataset=entry['case_id'][3:];saved=out/'V1'/(dataset+'_content_manifest.json')
            if not saved.exists():
                identities.append({'dataset':dataset,'status':'BLOCKED','reason':'V1 full content identity unavailable'})
                continue
            current=inspect_offline(dataset,data_root)['manifest']
            status='PASS' if current==json.loads(saved.read_text()) else 'FAIL'
            identities.append({'dataset':dataset,'status':status,'content_id':current['content_id']})
        details['V5_dataset_identity']=identities
        after=source_snapshot(root);save(out/'source-after.json',after)
        if before!=after:raise RuntimeError('Candidate source changed during fixed fixture execution')
        dependencies={n:importlib.util.find_spec(n) is not None for n in ('numpy','torch','torchvision','yaml')}
        if all(dependencies[n] for n in ('torch','torchvision','yaml')):
            from .run import assert_runtime_imports
            origins.update({n:r['path'] for n,r in assert_runtime_imports().items()})
            import_status='PASS'
        else:import_status='BLOCKED'
        if guard['optimizer_construction_attempts']!=0:raise RuntimeError('Nontraining optimizer guard triggered')
    counts={'V1datasets':len(v1['datasets']),'V1lookups':sum(e['lookups'] for e in v1['datasets']),
        'V2schedules':len(v2['cases']),'V3cases':len(v3),'V4cases':len(v4),'V5cases':len(v5),'stats':len(stats)}
    status='FAIL' if any(f['status']=='FAIL' for f in fixtures+identities) else (
        'PASS' if all(f['status']=='PASS' for f in fixtures+identities) and import_status=='PASS' and counts==COUNTS else 'BLOCKED')
    report={'schema':'scientific-nontraining-report-v2','status':status,'optimizer_steps':0,
        'optimizer_guard':guard,'counts':counts,'planned_counts':COUNTS,'fixtures':fixtures,
        'V5_dataset_identity':identities,'source_identity_before_after_equal':before==after,
        'full_runtime_import_status':import_status,'import_origins':origins,'dependencies':dependencies,
        'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},
        'seconds':time.perf_counter()-started,'integrated_attempts_executed':0,'claim_eligible':False,
        'blocked':[{'case_id':f['case_id'],'reason':f.get('reason',f.get('model_reason','See fixture details'))} for f in fixtures if f['status']=='BLOCKED']}
    save(out/'fixture-details.json',details);save(out/'report.json',report)
    return 0 if status=='PASS' else 3 if status=='BLOCKED' else 1


def v6(root,data_root,out,original_root):
    """LATER ONLY: fixed30 honest/common attempts,60rounds,max132SGD."""
    import torch
    if not torch.cuda.is_available():raise RuntimeError('V6 needs BOTH cpu and cuda; no fallback/no partial start')
    from .run import run_entry,_device_gate
    from .scenario import prepare
    from .protocol import DATASETS,METHODS,make_protocol
    from .instrumentation import StepBudget
    _device_gate('cuda')
    budget=StepBudget(132);attempts=[];pairing={}
    host_limit=min(24*1024**3,int(.75*physical_ram()))
    if len(DATASETS)*len(METHODS)*2!=30 or 30*2!=60 or 30*2*2+3*2*2!=132:
        raise RuntimeError('V6 frozen budget mismatch before execution')
    save(out/'plan.json',{'attempts':30,'planned_rounds':60,'max_client_sgd':120,'max_root_sgd':12,
        'max_total_sgd':132,'host_limit_bytes':host_limit,'claim_eligible':False})
    for dataset in DATASETS:
        for device in ('cpu','cuda'):
            scenario=out/('scenario-'+dataset+'-'+device)
            bundle=prepare(make_protocol(dataset,99001,purpose='v6',device=device),data_root,scenario)
            key=(bundle['rng']['namespace'],bundle['model_tensor_hash'])
            if dataset in pairing and pairing[dataset]!=key:raise RuntimeError('CPU/CUDA initial scenario identity mismatch')
            pairing[dataset]=key
            for method in METHODS:
                target=out/('attempt-'+dataset+'-'+device+'-'+method)
                certificate=run_entry(scenario,method,'common',device,target,step_budget=budget,
                    host_limit_bytes=host_limit,original_root=original_root,training_authorized=True)
                attempts.append({'path':str(target),'complete':certificate['complete'],'claim_eligible':False})
    from .records import validate_attempt
    for item in attempts:
        result=validate_attempt(item['path'])
        if not result['complete'] or not result['verified']:raise RuntimeError('V6 saved certificate invalid')
    if len(attempts)!=30 or budget.completed>132:raise RuntimeError('V6 bounds not met')
    save(out/'report.json',{'status':'PASS','attempts':attempts,'planned_rounds':60,
            'step_budget':budget.snapshot(),'claim_eligible':False,'affordability_established':False})
    return 0


def physical_ram():
    if sys.platform=='win32':
        from .windows_memory import physical_memory_bytes
        return physical_memory_bytes(failure_message='Cannot measure physical RAM')
    return os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('nontraining','v6'),required=True)
    parser.add_argument('--dataset-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-source',required=True)
    parser.add_argument('--original-root')
    parser.add_argument('--assert-zero-optimizer-steps',action='store_true')
    parser.add_argument('--acknowledge-training',action='store_true')
    args=parser.parse_args(argv)
    if args.suite=='nontraining' and (not args.assert_zero_optimizer_steps or args.acknowledge_training):
        parser.error('Nontraining requires zero optimizer assertion and forbids training acknowledgement')
    if args.suite=='v6' and (not args.acknowledge_training or args.assert_zero_optimizer_steps or not args.original_root):
        parser.error('Later V6 requires separate training acknowledgement and explicit original root')
    root=gate_source(args.expected_source)
    from ..comparison.provenance import safe_new_target
    protected=[root,Path(args.dataset_root)]
    if args.original_root:protected.append(Path(args.original_root))
    out=safe_new_target(args.output,protected=protected);out.mkdir(exist_ok=False)
    try:
        return nontraining(root,args.dataset_root,out) if args.suite=='nontraining' else v6(root,args.dataset_root,out,args.original_root)
    except BaseException as exc:
        save(out/'suite-failure.json',{'status':'FAIL','error':repr(exc),'completed':False,'claim_eligible':False})
        raise

if __name__=='__main__':raise SystemExit(main())
