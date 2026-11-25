"""Exercise backups offline using synthetic model files and tool calls."""
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_round2_archive as archive
from evaluate_experiment import judge
from run_crossed_seeds import CONFIG_KEYS, GENERATION_ARGS


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def make_example(name):
    return {'id': name, 'group_id': name, 'messages': [
        {'role': 'user', 'content': 'Get the clock.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'clock', 'arguments': {}}}]}],
        'tools': [{'function': {'name': 'clock', 'parameters': {'type': 'object', 'properties': {}}}}]}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.top = Path(temporary.name).resolve()
        self.root = self.top / 'run'
        self.primary = self.top / 'primary'
        self.dev = self.top / 'dev.jsonl'
        self.published = self.top / 'registered.json'
        self.witness = self.top / 'source.json'
        self.tag = 's42-L-short'
        self.checkpoint = self.tag + '-step1'
        self.start_patch = patch.object(archive, 'CORE', {self.tag})
        self.start_patch.start()
        self.addCleanup(self.start_patch.stop)
        self.sizes = patch.dict(archive.SPLITS, {'dev': 2, 'test': 3}, clear=True)
        self.sizes.start()
        self.addCleanup(self.sizes.stop)
        self.fixture()

    def fixture(self):
        tag, root = self.tag, self.root
        train = root / 'training' / tag
        checkpoint = train / 'checkpoint-1'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture-adapter')
        (train / 'last-checkpoint').symlink_to('checkpoint-1', target_is_directory=True)
        adapter = archive.digest(checkpoint / 'adapter_model.safetensors')
        pool = [{'id': f'p{i}'} for i in range(4000)]
        jsonl(self.primary / 'frozen/train.jsonl', pool)
        pool_sha = archive.digest(self.primary / 'frozen/train.jsonl')
        selected = root / 'selection/42/L-short.jsonl'
        jsonl(selected, pool[:1000])
        subset_sha = archive.digest(selected)
        archive.write(root / 'selection/manifest.json', {'cells': {tag: {'sha256': subset_sha}}})
        selection_sha = archive.digest(root / 'selection/manifest.json')
        self.goal = 3005228
        runtime = dict(cuda='fixture', gpu='fixture', trainable_parameter_count=1, token_goal=self.goal, starting_model_step=0)
        command = ['swift', '--model', '/fixture/model', '--external_plugins', '/fixture/experiment_callback.py',
                   '--dataset', 'subset', '--output_dir', 'target', '--seed', '42', '--data_seed', '42']
        run = dict(stage='target', seed=42, pool_sha256=pool_sha, data_sha256=subset_sha, command=command,
                   ids=[r['id'] for r in pool[:1000]], packages={'torch': 'fixture'}, sources_sha256={}, model_config_sha256='test')
        archive.write(train / 'run.json', run)
        archive.write(train / 'runtime.json', runtime)
        archive.write(train / 'completed.json', dict(final_step=1, answer_tokens=12, input_tokens=self.goal,
                                                      adapter_sha256={'1': adapter}))
        archive.write(train / 'exit.json', {'code': 0})
        archive.write(train / 'args.json', {})
        jsonl(train / 'steps.jsonl', [{'step': 1, 'input_tokens': self.goal}])
        locked = dict(label=tag, seed=42, method='L-short', checkpoint=self.checkpoint,
                      **archive.token_boundary(train, self.goal))
        archive.write(root / 'locks' / (tag + '.json'), locked)
        stage_paths = {'train': [train / s for s in ('completed.json', 'exit.json', 'args.json')]}
        scores, splits = {}, {}
        versions = {'python': '3.12.3 test', 'platform': 'Linux-test', 'vllm': 'fixture', 'torch': 'fixture'}
        for split, count in archive.SPLITS.items():
            path = self.dev if split == 'dev' else self.primary / 'frozen/test.jsonl'
            source = [make_example(f'{split}{i}') for i in range(count)]
            jsonl(path, source)
            directory = root / 'evaluation' / split / self.checkpoint
            config = {k: 'fixture' for k in CONFIG_KEYS}
            config.update(versions=versions, source_sha256=archive.digest(path), adapter_sha256={'adapter_model.safetensors': adapter},
                          args={**dict.fromkeys(GENERATION_ARGS), 'generate_only': False})
            splits[split] = {'generation': archive.gen_protocol(config)}
            archive.write(directory / 'config.json', config)
            raw = '<tool_call><function=clock></function></tool_call>'
            preds = [dict(id=r['id'], group_id=r['group_id'], raw_output=raw, **judge(r, raw)) for r in source]
            self.assertTrue(all(r['correct'] for r in preds))
            jsonl(directory / 'predictions.jsonl', preds)
            archive.write(directory / 'metrics.json', dict(count=count, correct=count, accuracy=1.0, reason_counts={'correct': count}))
            stage_paths[split] = [directory / s for s in ('config.json', 'metrics.json', 'predictions.jsonl')]
            scores[split] = {'count': count, 'correct': count, 'checkpoint': self.checkpoint}
        hashes, ledger = {}, []
        for stage, paths in stage_paths.items():
            task = stage + '-' + tag
            script = 'train_experiment.py' if stage == 'train' else 'evaluate_experiment.py'
            path = root / 'receipts' / (task + '.json')
            archive.write(path, dict(task=task, outputs={str(p): archive.digest(p) for p in paths},
                                     script_sha256=archive.digest(archive.SCRIPTS / script), elapsed_s=1))
            hashes[stage] = archive.digest(path)
            ledger.append(dict(event='done', task=task))
        jsonl(root / 'ledger.jsonl', ledger)
        archive.write(root / 'cells' / (tag + '.json'), dict(label=tag, seed=42, method='L-short', endpoint=locked,
                                                              scores=scores, receipt_sha256=hashes))
        archive.write(self.published, {'cell_order': [tag], 'training': {'pool_hash': pool_sha},
                                      'target_input_token_budget': self.goal, 'selection': {'manifest_sha256': selection_sha,
                                      'cells': {tag: {'sha256': subset_sha}}}})
        spec = {'cell_order': [tag], 'training': archive.train_protocol(run, runtime), 'splits': splits,
                'selections': {tag: {'sha256': subset_sha}}, 'pool_sha256': pool_sha, 'source_sha256': {},
                'public_registration_sha256': archive.digest(self.published), 'selection_manifest_sha256': selection_sha}
        archive.write(root / 'protocol-lock.json', {'spec': spec})
        self.protocol_sha = archive.digest(root / 'protocol-lock.json')
        environments = {
            'training': dict(active={'torch': 'fixture'}, reported={'torch': 'fixture'}, cuda='fixture', torch_import='fixture'),
            'evaluation': dict(active={'torch': 'fixture', 'vllm': 'fixture'}, torch_import='fixture',
                               python=versions['python'], platform=versions['platform']),
        }
        archive.write(root / 'sessions/1.json', dict(protocol_lock_sha256=self.protocol_sha, observed_eval_versions=versions,
                                                      environments=environments, observed_training_command=spec['training']['command']))
        archive.write(root / 'execution.json', dict(status='complete', planned=[tag], completed=[tag], missing=[]))
        archive.write(root / 'locked-models.json', {tag: self.checkpoint})
        archive.write(self.witness, archive.inventory(root))

    def verify(self, root):
        return archive.verify(root, self.witness, archive.digest(self.witness), self.protocol_sha,
                              self.published, self.primary, self.dev)

    def test_relocated_backup_independently_regrades_outputs(self):
        copy = self.top / 'relocated'
        shutil.copytree(self.root, copy, symlinks=True)
        proof = self.verify(copy)
        self.assertEqual(proof['regraded_predictions'], {'dev': 2, 'test': 3})
        self.assertEqual(proof['completed_tasks'], 3)
        (copy / 'training' / self.tag / 'checkpoint-1/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differ from the source'):
            self.verify(copy)

    def test_matching_archive_and_receipts_do_not_substitute_for_regrading(self):
        directory = self.root / 'evaluation' / 'test' / self.checkpoint
        path = directory / 'predictions.jsonl'
        predictions = archive.rows(path)
        predictions[0]['raw_output'] = 'unparseable response'
        jsonl(path, predictions)
        receipt = self.root / 'receipts' / ('test-' + self.tag + '.json')
        changed = archive.read(receipt)
        changed['outputs'][str(path)] = archive.digest(path)
        archive.write(receipt, changed)
        cell = self.root / 'cells' / (self.tag + '.json')
        changed = archive.read(cell)
        changed['receipt_sha256']['test'] = archive.digest(receipt)
        archive.write(cell, changed)
        archive.write(self.witness, archive.inventory(self.root))
        with self.assertRaisesRegex(ValueError, 'verdict'):
            self.verify(self.root)

    def test_unfinished_queue_and_foreign_directory_are_rejected(self):
        state = self.root / 'execution.json'
        closed = archive.read(state)
        archive.write(state, dict(closed, status='running'))
        with self.assertRaisesRegex(ValueError, 'not closed'):
            archive.inventory(self.root)
        archive.write(state, closed)
        (self.root / 'evaluation' / 'holdout-v2').mkdir()
        with self.assertRaisesRegex(ValueError, 'legacy'):
            archive.inventory(self.root)

    def test_separate_evaluations_are_outside_terminal_core_archive(self):
        for name in ('bfcl-live', 'holdout-evaluation'):
            sub = self.root / name
            sub.mkdir()
            (sub / 'prospective-private-input').write_bytes(b'synthetic')
        self.assertEqual(archive.read(self.witness), archive.inventory(self.root))

    def test_shadowed_package_metadata_requires_valid_active_imports(self):
        path = self.root / 'sessions/1.json'
        session = archive.read(path)
        session['environments']['training']['torch_import'] = 'altered'
        archive.write(path, session)
        archive.write(self.witness, archive.inventory(self.root))
        with self.assertRaisesRegex(ValueError, 'PyTorch import'):
            self.verify(self.root)

    def test_session_cannot_change_sampling_or_eval_versions(self):
        path = self.root / 'sessions/1.json'
        session = archive.read(path)
        session['observed_eval_versions']['vllm'] = 'wrong version'
        archive.write(path, session)
        archive.write(self.witness, archive.inventory(self.root))
        with self.assertRaisesRegex(ValueError, 'package versions'):
            self.verify(self.root)


if __name__ == '__main__':
    unittest.main()
