r"""Inventory a closed Round 2 core run and independently verify its relocated backup.

On the source host, after the training controller exits:
  python scripts/verify_round2_archive.py inventory --root local/followup/round2 --out local/round2-source.json
On the backup host, after copying the source inventory and complete run directory:
  python scripts/verify_round2_archive.py verify --root local/runpod/repo/local/followup/round2 \
    --inventory local/round2-source.json --inventory-sha256 SOURCE_HASH --protocol-sha256 PRETRAINING_LOCK_HASH \
    --registration results/followup/round2-registration.json --primary local/runpod/repo/local/course-run \
    --dev local/prepared/dev.jsonl --out local/round2-backup-proof.json

The proof covers target training and legacy dev/test only. It contains hashes and
counts, not metric values, requests or predictions. Outputs must be outside the archive.
"""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

from difficulty_admission import replay
from run_course_experiment import digest, read, rows, write
from run_crossed_seeds import gen_protocol, token_boundary
from run_followup import SPLITS, STAGES, completed_cells, train_protocol


CORE = {f"s{s}-{m}" for s in (45, 46) for m in ("A", "B", "C", "E2", "E2b")} | {
    f"s{s}-L-{size}" for s in range(42, 47) for size in ("short", "long")}
FORMAT = "round2-core-archive-v1"
SCRIPTS = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(path):
    """Reject prospective paths, including aliases, before opening any file."""
    for candidate in (path.absolute(), path.resolve()):
        for part in candidate.parts:
            slug = part.casefold().replace('_', '-').replace(' ', '-')
            require('holdout-v2' not in slug and 'holdoutv2' not in slug, 'Prospective artifacts are outside this audit')
    return path


def terminal(root):
    safe_path(root)
    lock = read(safe_path(root / 'protocol-lock.json'))['spec']
    order = lock['cell_order']
    require(len(order) == len(CORE) and set(order) == CORE, 'Expected the 20 registered core cells')
    status = read(safe_path(root / 'execution.json'))
    require(status['status'] == 'complete' and status['planned'] == order and not status['missing']
            and len(status['completed']) == len(CORE) and set(status['completed']) == CORE,
            'The core queue has not closed all 20 cells')
    return lock


def inventory(root):
    root = safe_path(root).resolve()
    terminal(root)
    files, links = {}, {}
    for folder, dirs, names in os.walk(root, followlinks=False):
        folder = Path(folder)
        if folder == root:
            dirs[:] = [name for name in dirs if name not in {'bfcl-live', 'holdout-evaluation'}]
        if folder == root / 'evaluation':
            require(set(dirs) <= set(SPLITS), 'Only legacy dev/test predictions may enter this archive')
        for name in sorted(dirs + names):
            path = safe_path(folder / name)
            relative = path.relative_to(root).as_posix()
            archived_attempt = Path(relative).parts[0] == 'partial-attempts'
            require(name not in {'.git', '__pycache__'} and (archived_attempt or not name.endswith(('.tmp', '.partial'))),
                    f'Unexpected cache or unfinished transfer: {relative}')
            if path.is_symlink():
                target = path.resolve(strict=True)
                require(target.is_relative_to(root), f'Archive alias escapes the run: {relative}')
                require(((folder.parent == root / 'training' and folder.name in CORE) or archived_attempt) and name == 'last-checkpoint'
                        and target.parent == folder and target.name.startswith('checkpoint-') and target.is_dir(),
                        f'Unexpected archive alias: {relative}')
                links[relative] = target.relative_to(root).as_posix()
            elif name in names:
                require(path.is_file(), f'Unexpected special file: {relative}')
                files[relative] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return {'format': FORMAT, 'files': files, 'symbolic_links': links}


def indexed(path, size, expected_hash):
    safe_path(path)
    require(digest(path) == expected_hash, f'Frozen input digest differs: {path.name}')
    data = rows(path)
    result = {str(r['id']): r for r in data}
    require(len(data) == len(result) == size, f'Wrong number of unique rows: {path.name}')
    return result


def sessions(root, spec):
    """Only grant the host metadata and path variations recorded by the controller."""
    training_variants, generation_variants = [], []
    for path in sorted((root / 'sessions').glob('*.json')):
        observed = read(path)
        require(observed['protocol_lock_sha256'] == digest(root / 'protocol-lock.json'),
                'Session references a different protocol lock')
        train = deepcopy(spec['training'])
        original = train['command']
        actual = observed['observed_training_command']
        require(len(original) == len(actual), 'Session training command length changed')
        for flag in ('--model', '--external_plugins'):
            index = original.index(flag) + 1
            require(Path(actual[index]).is_absolute() and Path(original[index]).name == Path(actual[index]).name,
                    f'Unexpected name for relocated {flag}')
            original[index] = actual[index]
        require(original == actual, 'Training flags changed across sessions')
        envs = observed['environments']
        for role, environment in envs.items():
            if role == 'training':
                pinned_packages = spec['training']['packages']
                require(environment['cuda'] == train['runtime']['cuda'] and 'torch' in environment['reported'],
                        'Training CUDA build or runtime report changed')
                train['packages'] = environment['reported']
            else:
                require(role == 'evaluation', 'Unrecognized session environment')
                versions = spec['splits']['dev']['generation']['versions']
                pinned_packages = {k: v for k, v in versions.items() if k not in ('python', 'platform')}
            require(set(environment['active']) == set(pinned_packages), 'Wrong pinned package coverage')
            for package, version in pinned_packages.items():
                active = environment['active'][package]
                if package == 'torch' and role == 'training':
                    active, version = active.split('+')[0], version.split('+')[0]
                require(active == version, f'{role}: imported package version changed for {package}')
            require(environment['torch_import'].split('+')[0] == pinned_packages['torch'].split('+')[0],
                    f'{role}: PyTorch import changed')
        require(set(envs) == {'training', 'evaluation'}, 'Missing environment probes')
        versions = observed['observed_eval_versions']
        require(all(versions[k] == envs['evaluation'][k] for k in ('python', 'platform')),
                'Evaluation runtime observation differs')
        both_splits = {}
        for split in SPLITS:
            generation = deepcopy(spec['splits'][split]['generation'])
            pinned = generation['versions']
            require(versions['python'].split()[0] == pinned['python'].split()[0], 'Evaluation Python version changed')
            pinned.update(python=versions['python'], platform=versions['platform'])
            require(pinned == versions, 'Evaluation package versions changed')
            both_splits[split] = generation
        training_variants.append(train)
        generation_variants.append(both_splits)
    require(bool(training_variants), 'Missing execution-session provenance')
    return training_variants, generation_variants


def audit_cell(root, tag, checkpoint, spec, sources, pool, observed):
    training_variants, generation_variants = observed
    seed_part, method = tag.split('-', 1)
    seed = int(seed_part[1:])
    folder = root / 'training' / tag
    cell = read(root / 'cells' / (tag + '.json'))
    goal = spec['training']['runtime']['token_goal']
    require(goal == 3_005_228, f'{tag}: incorrect core token budget')
    bound = token_boundary(folder, goal)
    expected = dict(label=tag, seed=seed, method=method, checkpoint=f"{tag}-step{bound['step']}", **bound)
    require(checkpoint == expected['checkpoint'] and cell['endpoint'] == expected
            and (cell['label'], cell['seed'], cell['method']) == (tag, seed, method), f'{tag}: endpoint mismatch')
    run, runtime = read(folder / 'run.json'), read(folder / 'runtime.json')
    require(run['stage'] == 'target' and run['seed'] == seed and run['pool_sha256'] == spec['pool_sha256']
            and read(folder / 'exit.json')['code'] == 0 and train_protocol(run, runtime) in training_variants,
            f'{tag}: training settings or exit code differ')
    selection = root / 'selection' / str(seed) / (method + '.jsonl')
    require(digest(selection) == run['data_sha256'] == spec['selections'][tag]['sha256'], f'{tag}: wrong subset hash')
    samples = rows(selection)
    ids = [str(row['id']) for row in samples]
    require(len(samples) == len(set(ids)) == 1000 and ids == list(map(str, run['ids']))
            and all(row == pool.get(key) for row, key in zip(samples, ids)), f'{tag}: selected examples differ')
    for stage in STAGES:
        receipt = read(root / 'receipts' / f'{stage}-{tag}.json')
        script = 'train_experiment.py' if stage == 'train' else 'evaluate_experiment.py'
        require(receipt['script_sha256'] == digest(SCRIPTS / script) and receipt['elapsed_s'] > 0,
                f'{tag}: {stage} script or elapsed time differs')
    for split, count in SPLITS.items():
        folder = root / 'evaluation' / split / checkpoint
        config = read(folder / 'config.json')
        controls = spec['splits'][split]['generation']
        require(any(gen_protocol(config) == version[split] for version in generation_variants)
                and config['adapter_sha256']['adapter_model.safetensors'] == bound['final_adapter_sha256'],
                f'{tag}: {split} generation settings or adapter differ')
        replay(folder, sources[split], controls['source_sha256'])
        metrics = read(folder / 'metrics.json')
        require(cell['scores'][split] == {'count': count, 'correct': metrics['correct'], 'checkpoint': checkpoint},
                f'{tag}: {split} cell result differs from regraded outputs')


def verify(root, witness, witness_sha, protocol_sha, registration, primary, dev):
    root = safe_path(root).resolve()
    require(digest(safe_path(root / 'protocol-lock.json')) == protocol_sha, 'Private lock differs from the pretraining pin')
    require(digest(safe_path(witness)) == witness_sha, 'Unrecognized source inventory')
    expected = read(witness)
    require(expected.get('format') == FORMAT, 'Unsupported inventory')
    local = inventory(root)
    for key in ('files', 'symbolic_links'):
        require(local[key] == expected[key], f'Local {key} differ from the source inventory')
    spec = terminal(root)
    require(digest(safe_path(registration)) == spec['public_registration_sha256'], 'Public registration differs')
    published = read(registration)
    require(spec['cell_order'] == published['cell_order'] and spec['pool_sha256'] == published['training']['pool_hash']
            and spec['training']['runtime']['token_goal'] == published['target_input_token_budget'], 'Registration protocol differs')
    require(digest(root / 'selection/manifest.json') == spec['selection_manifest_sha256'] == published['selection']['manifest_sha256']
            and all(spec['selections'][tag]['sha256'] == published['selection']['cells'][tag]['sha256'] for tag in CORE),
            'Selection master or subset digests differ')
    for name, expected in spec['source_sha256'].items():
        require(Path(name).name == name and digest(SCRIPTS / name) == expected, f'Frozen controller source differs: {name}')
    for key in ('cells', 'locks'):
        require({p.stem for p in (root / key).glob('*.json')} == CORE, f'Unexpected {key} coverage')
    locks = completed_cells(SimpleNamespace(root=root), spec)
    require(set(locks) == CORE and locks == read(root / 'locked-models.json'), 'Final endpoint map differs')
    tasks = {f'{stage}-{tag}' for tag in CORE for stage in STAGES}
    require(tasks == {p.stem for p in (root / 'receipts').glob('*.json')}, 'Unexpected completed task receipt set')
    done = Counter(r['task'] for r in rows(root / 'ledger.jsonl') if r.get('event') == 'done')
    require(done == Counter(tasks), 'Missing, duplicated or unexpected completed ledger stages')
    pool = indexed(primary / 'frozen/train.jsonl', 4000, spec['pool_sha256'])
    paths = {'dev': dev, 'test': primary / 'frozen/test.jsonl'}
    sources = {split: indexed(paths[split], count, spec['splits'][split]['generation']['source_sha256'])
               for split, count in SPLITS.items()}
    observed = sessions(root, spec)
    for tag in spec['cell_order']:
        audit_cell(root, tag, locks[tag], spec, sources, pool, observed)
    return {'terminal_state': 'complete', 'completed_cells': len(locks), 'completed_tasks': len(tasks),
            'regraded_predictions': {split: count * len(locks) for split, count in SPLITS.items()},
            'file_count': len(local['files']), 'bytes': sum(v['bytes'] for v in local['files'].values()),
            'public_registration_sha256': digest(registration), 'protocol_lock_sha256': digest(root / 'protocol-lock.json'),
            'locked_models_sha256': digest(root / 'locked-models.json'), 'source_inventory_sha256': witness_sha,
            'verifier_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('inventory', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--inventory-sha256')
    parser.add_argument('--protocol-sha256', help='SHA-256 recorded when the pretraining protocol was frozen')
    parser.add_argument('--registration', type=Path, default=Path('results/followup/round2-registration.json'))
    parser.add_argument('--primary', type=Path, default=Path('local/runpod/repo/local/course-run'))
    parser.add_argument('--dev', type=Path, default=Path('local/prepared/dev.jsonl'))
    args = parser.parse_args()
    root = safe_path(args.root).resolve()
    output = safe_path(args.out).resolve()
    require(not output.is_relative_to(root), 'Write proof outside the audited archive')
    if args.mode == 'inventory':
        proof = inventory(root)
    else:
        if not args.inventory or not args.inventory_sha256 or not args.protocol_sha256:
            parser.error('verify requires --inventory, --inventory-sha256 and --protocol-sha256')
        require(output != safe_path(args.inventory).resolve(), 'Keep source inventory separate from the local proof')
        proof = verify(root, args.inventory, args.inventory_sha256, args.protocol_sha256, args.registration, args.primary, args.dev)
    write(output, proof)
    print(output.name, digest(output))


if __name__ == '__main__':
    main()
