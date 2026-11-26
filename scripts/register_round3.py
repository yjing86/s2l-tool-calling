"""Bind the third-round selection and previously scored controls before training."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def registration(archive, subsets):
    scores = archive / 'course-run/scores/step-0.jsonl'
    pool = rows(scores)
    tokens = {str(x['id']): x['value_tokens'] for x in pool}
    total = sum(x['input_tokens'] for x in pool)
    assert len(tokens) == len(pool) == 4000
    budgets = {str(size): math.ceil(2*size*total/len(pool)) for size in (400,1000)}
    sampled = {}
    for seed in range(42,47):
        for m in 'ABC':
            p = subsets / str(seed) / f'{m}.jsonl'
            prior = (archive / ('course-run/selection' if seed < 45 else 'followup/round2/selection') / str(seed) / p.name)
            data = rows(p)
            ids = {str(x['id']) for x in data}
            assert len(ids) == len(data) == 400 and ids <= tokens.keys()
            sampled[f's{seed}-{m}'] = dict(sha256=sha(p), size=len(ids),
                mean_value_tokens=statistics.mean(tokens[i] for i in ids),
                functions=len({x['messages'][-1]['tool_calls'][0]['function']['name'] for x in data}),
                overlap_with_25pct=len(ids & {str(x['id']) for x in rows(prior)}))
    full = archive/'full-reference/training'
    ref = {}
    for name,size in [('full10',400),('full25',1000)]:
        step=next(x['step'] for x in rows(full/'steps.jsonl') if x['input_tokens']>=budgets[str(size)])
        adapter=full/f'checkpoint-{step}'
        ref[name]={'step':step,'weights_sha256':sha(adapter/'adapter_model.safetensors'),'config_sha256':sha(adapter/'adapter_config.json')}
    return dict(seeds=list(range(42,47)),pool_examples=4000,pool_tokens=total,
        target_input_tokens=budgets, frozen_pool_sha256=sha(archive/'course-run/frozen/train.jsonl'),
        subsets=sampled,historical_full_seed42=ref,
        target='Qwen3.5-2B; Round 2 LoRA, constant schedule with five warmup steps, microbatch 4, effective batch 16; complete reference-call objective',
        selection='Frozen 0.8B proxy, steps 0/10/30/80/250; select_experiment.py with size=400, k=10, cluster seed 42; paired selection and training seeds',
        planned_fits='4 full-pool fits seeds 43-46, saving first optimizer crossings of both budgets; 15 subset fits (A,B,C x 5 seeds) saving first crossing of the smaller budget',
        evaluation='Reuse Round 2 prompts, greedy scorer and 1500 previously examined holdout-v2 requests. Freeze all weights before new inference. No development or legacy testing.',
        historical_holdout_sha256=sha(Path('results/followup/round2-holdout.json')),
        holdout_sha256=sha(archive/'followup/holdout-v2/test.jsonl'),
        primary=['full25-A25','C10-A10'],primary_test='Two-sided paired seed t tests, df=4, Holm familywise alpha 0.05',
        secondary=['B10-A10','C10-B10','full10-A10','(C10-A10)-(C25-A25)'],
        uncertainty='Report each seed delta, two-sided 95% seed t interval, fixed-model paired request-group bootstrap with 10000 draws and seed 260205',
        noise='Both previously measured splits; sample SD, enumerate all ten triples, noncentral t planning at one-sided alpha .05, power .8 for 1 and 2 points; precision curve at observed C-B SD. Historical triple frequencies are descriptive.',
        interpretation='Follow-up on a previously examined test set. No equivalence claim from nonsignificance, and budget-specific superiority requires a comparison of effects.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('subsets',type=Path)
    a=p.parse_args()
    print(json.dumps(registration(a.archive,a.subsets),indent=2))
