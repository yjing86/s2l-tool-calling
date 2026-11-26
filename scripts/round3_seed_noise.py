"""Describe seed variation and plan paired tests using observed sample SDs."""
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from scipy.stats import nct, t
from matplotlib import pyplot as plt


def required_n(sd, delta):
    for n in range(2,10001):
        if nct.sf(t.ppf(.95,n-1), n-1, delta*n**.5/sd) >= .8:
            return n
    raise ValueError("Sample size exceeds search range")


def main():
    dst=Path('results/followup')
    old=json.loads((dst/'round2-legacy.json').read_text())['splits']['test']['methods']
    new=json.loads((dst/'round2-holdout.json').read_text())['endpoints']
    seeds=list(range(42,47))
    methods=['A','B','C','E2','E2b','L-short','L-long']
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    summary={}
    for split in ['original','reserved']:
        scores={m:[(old[m]['by_seed'][str(s)] if split=='original' else new[f's{s}-{m}'])['accuracy_pct'] for s in seeds] for m in methods}
        d={}
        for left,right in [('C','B'),('B','A'),('C','A')]:
            diffs=[a-b for a,b in zip(scores[left],scores[right])]
            sd=stdev(diffs)
            d[left+'-'+right]={'differences_pp':diffs,'sample_sd_pp':sd,
                              'planned_n':{str(k):required_n(sd,k) for k in [1,2]}}
        vals=d['C-B']['differences_pp']
        triples=[dict(seeds=[seeds[i] for i in ix],mean_pp=mean(vals[i] for i in ix),all_positive=all(vals[i]>0 for i in ix)) for ix in combinations(range(5),3)]
        summary[split]=dict(method_sd_pp={m:stdev(v) for m,v in scores.items()},contrasts=d,triples=triples,
                            all_positive_triples=sum(x['all_positive'] for x in triples))
        ns=np.arange(3,41)
        ax.plot(ns,t.ppf(.975,ns-1)*stdev(vals)/np.sqrt(ns),label=split.title()+' test')
    ax.axhline(1,color='black',linestyle=':',label='1 percentage point')
    ax.set(xlabel='Paired seeds',ylabel='Estimated 95% interval half-width (pp)',title='Precision of C minus B at the observed seed variation')
    ax.legend()
    fig.savefig('docs/figures/round3-seed-precision.png',dpi=160)
    (dst/'round3-seed-noise.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    main()
