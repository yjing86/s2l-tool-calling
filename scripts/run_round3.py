"""Run all registered fits, freeze the panel, then generate on the reused reserve."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from register_round3 import sha, rows
from round2_holdout import task_args
from verify_tool_call_loss import swift_row


def read(p):
    return json.loads(p.read_text())


def write(p,v):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n')
    tmp.replace(p)


def fit(a, seed, method, reg):
    full=method=='full'
    tag=f's{seed}-{method}'
    pool=a.inputs/'train.jsonl'
    data=pool if full else a.inputs/'selection'/str(seed)/f'{method}.jsonl'
    assert sha(data)==(reg['frozen_pool_sha256'] if full else reg['subsets'][tag]['sha256'])
    final=a.out/'training'/tag
    final.parent.mkdir(exist_ok=True)
    if not final.exists():
        out=Path(tempfile.mkdtemp(prefix=tag+'-',dir=a.out/'attempts'))
        items=rows(data)
        swift_data=out/'train.swift.jsonl'
        with swift_data.open('w') as f:
            for row in items:
                print(json.dumps(swift_row(row),ensure_ascii=False),file=f)
        ref=read(a.inputs/'reference-run.json')
        cmd=ref['command'].copy()
        cmd[0]=str(a.train_python.parent/'swift')
        changes={'--model':a.model,'--dataset':swift_data,'--output_dir':out,'--seed':seed,'--data_seed':seed,
                 '--external_plugins':Path(__file__).with_name('round3_hook.py' if full else 'experiment_callback.py').resolve(),
                 '--callbacks':'round3_full' if full else 'tool_experiment'}
        for flag,v in changes.items():
            cmd[cmd.index(flag)+1]=str(v)
        meta=dict(ref,command=cmd,stage='target',model=str(a.model),seed=seed,ids=[str(x['id']) for x in items],
                  pool=str(pool),pool_sha256=sha(pool),data_sha256=sha(data),planned_steps_one_epoch=(len(items)+15)//16,save_steps=[])
        write(out/'run.json',meta)
        env=dict(os.environ,TOOL_TRAIN_MANIFEST=str(out/'run.json'),PATH=str(a.train_python.parent)+':'+os.environ['PATH'],
                 PYTHONPATH=str(Path(__file__).resolve().parent)+':'+os.environ.get('PYTHONPATH',''))
        with (out/'console.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        if not (out/'completed.json').is_file():
            raise RuntimeError(f'Missing completion receipt for {tag}')
        out.rename(final)
    completed=read(final/'completed.json')
    steps=rows(final/'steps.jsonl')
    endpoints={}
    for size in ([400,1000] if full else [400]):
        budget=reg['target_input_tokens'][str(size)]
        first=next(x for x in steps if x['input_tokens']>=budget)
        step=first['step']
        path=final/f'checkpoint-{step}'
        digest=sha(path/'adapter_model.safetensors')
        assert completed['adapter_sha256'][str(step)]==digest
        if size==(1000 if full else 400):
            assert completed['final_step']==step
        label=f's{seed}-{method}{25 if size==1000 else 10}'
        endpoints[label]=dict(adapter=str(path),weights_sha256=digest,config_sha256=sha(path/'adapter_config.json'),
                              step=step,input_tokens=first['input_tokens'],examples=first['examples'])
    print('Verified fit',tag,flush=True)
    return endpoints


def run(a):
    reg=read(Path(__file__).resolve().parents[1]/'results/followup/round3-registration.json')
    assert sha(a.inputs/'train.jsonl')==reg['frozen_pool_sha256']
    if read(a.inputs/'reference-run.json')['batch']!=16:
        raise ValueError('Expected frozen target optimization settings')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'attempts').mkdir(exist_ok=True)
    panel={}
    for seed,method in [(s,'full') for s in range(43,47)]+[(s,m) for s in range(42,47) for m in 'ABC']:
        panel.update(fit(a,seed,method,reg))
    old=reg['historical_full_seed42']['full10']
    hist=a.inputs/'full10-s42'
    assert sha(hist/'adapter_model.safetensors')==old['weights_sha256'] and sha(hist/'adapter_config.json')==old['config_sha256']
    panel['s42-full10']={**old,'adapter':str(hist)}
    assert len(panel)==24
    freeze=a.out/'freeze.json'
    if freeze.exists():
        assert read(freeze)==panel
    else:
        write(freeze,panel)
    print('All adapters frozen:',sha(freeze),flush=True)
    if a.train_only:
        return
    assert sha(a.inputs/'test.jsonl')==reg['holdout_sha256']
    assert sha(a.inputs/'prompts.jsonl')==read(a.inputs/'holdout-provenance.json')['prompts_sha256']
    dest=a.out/'predictions'
    dest.mkdir(exist_ok=True)
    for tag,rec in panel.items():
        final=dest/tag
        if not final.exists():
            tmp=Path(tempfile.mkdtemp(prefix=tag+'-inference-',dir=a.out/'attempts'))
            cmd=task_args(a.inputs/'test.jsonl',a.model,tmp,prompts=a.inputs/'prompts.jsonl',adapter=Path(rec['adapter']))
            with (tmp/'console.log').open('w') as log:
                subprocess.run([str(a.eval_python),*cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(tmp/'metrics.json')['count']==1500
            tmp.rename(final)
        assert len(rows(final/'predictions.jsonl'))==1500
        print('Verified generation',tag,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['inputs','out','model','train-python','eval-python']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--train-only',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
