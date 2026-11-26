"""Regrade the new panel and pair it with frozen Round 2 outcomes on the same requests."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from round2_holdout import audit_predictions, inspect_released, read, fingerprint
from round2_holdout_stats import _bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','historical','data','out']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    old=read(a.historical/'freeze.json')
    historic=read(a.historical/'summary.json')
    reg=read(Path(__file__).resolve().parents[1]/'results/followup/round3-registration.json')
    assert fingerprint(a.historical/'summary.json')==reg['historical_holdout_sha256']
    refs,groups=inspect_released(a.data,old)
    ids=sorted(refs)
    panel=read(a.run/'freeze.json')
    seeds=reg['seeds']
    expected={f's{s}-{m}10' for s in seeds for m in ['A','B','C','full']}|{f's{s}-full25' for s in seeds[1:]}
    assert panel.keys()==expected
    hist_by_tag={e['id']:e for e in old['panel']}
    models={}
    commits={}
    def audit(tag, endpoint, path):
        result=audit_predictions(path, endpoint,old,historic['provenance']['prompts_sha256'],refs)
        commits[tag]=fingerprint(path/'predictions.jsonl')
        models[tag]=np.array([result[i] for i in ids],dtype=float)
    for tag,entry in panel.items():
        audit(tag,dict(id=tag,adapter_sha256={'adapter_config.json':entry['config_sha256'],'adapter_model.safetensors':entry['weights_sha256']}),a.run/'predictions'/tag)
    for s in seeds:
        for method in ['A','C']+(['full'] if s==42 else []):
            tag='full-s42-step125' if method=='full' else f's{s}-{method}'
            dest=f's{s}-{method}25'
            audit(dest,hist_by_tag[tag],a.historical/'predictions'/tag)
            assert commits[dest]==historic['provenance']['predictions_sha256'][tag]
    effects={'full25-A25':{'full25':1,'A25':-1},'C10-A10':{'C10':1,'A10':-1},
             'B10-A10':{'B10':1,'A10':-1},'C10-B10':{'C10':1,'B10':-1},'full10-A10':{'full10':1,'A10':-1},
             '(C10-A10)-(C25-A25)':{'C10':1,'A10':-1,'C25':-1,'A25':1}}
    result={}
    columns=[]
    for label,weights in effects.items():
        paired=np.array([100*sum(w*models[f's{s}-{m}'] for m,w in weights.items()) for s in seeds])
        d=paired.mean(axis=1); avg=float(d.mean()); se=float(d.std(ddof=1)/len(d)**.5)
        half=t.ppf(.975,4)*se
        rec=dict(seed_differences_pp=d.tolist(),mean_pp=avg,seed_ci95=[avg-half,avg+half])
        if label in reg['primary']:
            rec['two_sided_p']=float(2*t.sf(abs(avg/se),4)) if se else (0.0 if avg else 1.0)
        result[label]=rec
        columns.append(paired.mean(axis=0))
    # Round 2 used this fixed bootstrap seed. Verify before reusing the implementation.
    import round2_holdout_stats as stats
    assert stats.BOOTSTRAP_SEED==260205 and stats.BOOTSTRAP_DRAWS==10000
    intervals=_bootstrap(np.column_stack(columns),[groups[i] for i in ids])
    for j,k in enumerate(effects):
        result[k]['request_ci95']=intervals[:2,j].tolist()
    keys=sorted(reg['primary'],key=lambda k:result[k]['two_sided_p'])
    cumulative=0.
    for j,k in enumerate(keys):
        cumulative=max(cumulative,min(1.,(2-j)*result[k]['two_sided_p']))
        result[k]['holm_p']=cumulative
    public=dict(unique_requests=len(ids),new_endpoints=len(panel),total_new_predictions=len(panel)*len(ids),
                correct={tag:int(arr.sum()) for tag,arr in models.items()},contrasts=result,
                prediction_sha256=commits,model_freeze_sha256=fingerprint(a.run/'freeze.json'))
    a.out.write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    main()
