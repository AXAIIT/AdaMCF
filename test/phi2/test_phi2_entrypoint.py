"""总入口回归：真实小型两阶段、旧入口数值一致、失败保存和控制器路径。"""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


# 单独运行或 unittest discover 时，生产模块统一从模型目录导入。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHI2_DIR = PROJECT_ROOT / 'test' / 'pretrain_phi2'
for directory in (PROJECT_ROOT, PHI2_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

import pretrain_phi2_pruning as entry
import pretrain_phi2_pruning as controller
from finetune_phi2_lora import RecoveryConfig
from only_train_once import OTO
from phi2_data import file_sha256
from test_phi2_contract import prepare
from prune_phi2_hesso_cric import PruningConfig, main as pruning_cli, run as legacy_run
from test_phi2_contract import tiny_model
from test_phi2_training import tokenizer


def fixture(root, mode='joint', fmt='safetensors'):
    base = root / 'model'
    base.mkdir()
    tiny_model().save_pretrained(base)
    tokenizer().save_pretrained(base)
    source = root / 'synthetic_examples.json'
    source.write_text(json.dumps([
        {'instruction': f'w{i}', 'input': 'w2', 'output': 'w3 w4 w5'}
        for i in range(20)]), encoding='utf-8')
    data = root / 'data'
    prepare(SimpleNamespace(source=source, model_dir=base, output=data, seed=42,
                            max_length=32, train_samples=10, alpha_samples=6, eval_samples=4))
    config = PruningConfig(
        pruning_mode=mode, export_format=fmt, warmup_steps=0,
        target_group_sparsity=.25, sampling_steps=2, max_cycle_period=2,
        hybrid_training_steps=3, post_pruning_steps=0, alpha_eval_batches=2,
        log_every=100, recovery=asdict(RecoveryConfig(
            enabled=True, steps=2, gradient_accumulation_steps=2,
            rank=2, dropout=0, eval_every=1, log_every=1)))
    path = root / 'config.json'
    path.write_text(json.dumps(asdict(config)), encoding='utf-8')
    return SimpleNamespace(config=path, model_dir=base, data_dir=data, output=root / 'result',
                           device='cpu', export_format=None, log_every=None)


class PhiEntrypointTests(unittest.TestCase):
    def test_real_two_stage_modes_formats_and_legacy_numerical_parity(self):
        cases = [('joint', 'safetensors'), ('joint', 'pt'), ('joint', 'onnx'),
                 ('num_head', 'pt'), ('head_dim', 'safetensors'), ('ffn', 'pt')]
        with tempfile.TemporaryDirectory() as temp:
            for number, (mode, fmt) in enumerate(cases):
                with self.subTest(mode=mode, format=fmt):
                    root = Path(temp) / str(number)
                    root.mkdir()
                    capture = io.StringIO()
                    with contextlib.redirect_stdout(capture):
                        args = fixture(root, mode, fmt)
                        before = file_sha256(args.config)
                        result = entry.main(args)
                    text = capture.getvalue()
                    for i in range(1, 7):
                        self.assertIn(f'第{i}步：', text)
                    self.assertIn('阶段一', text)
                    self.assertIn('阶段二', text)
                    self.assertIn('最终模型：', text)
                    self.assertTrue(result['two_stage'])
                    self.assertFalse(result['full_benchmark_evaluated'])
                    self.assertTrue(result['recovery']['base_parameters_unchanged'])
                    self.assertTrue(result['recovery']['merged_reload_verified'])
                    self.assertEqual(before, file_sha256(args.config))
                    self.assertFalse((args.output / 'stage1_pruning/lora_recovery').exists())
                    saved = json.loads((args.output / 'pruning_config.json').read_text())
                    self.assertFalse(saved['recovery']['enabled'])
                    self.assertTrue((args.output / 'stage2_finetuning/adapter').is_dir())
                    self.assertEqual(result['recovery']['export_format'], fmt)
                    self.assertEqual(json.loads((args.output / 'status.json').read_text())['state'], 'completed')
                    self.assertIn('内部响应 NLL', (args.output / 'summary.md').read_text())
                    threads = torch.get_num_threads()
                    try:
                        with patch('sys.argv',['prune','--verify-only','--output',str(args.output/'stage1_pruning'),
                                               '--device','cpu']), contextlib.redirect_stdout(io.StringIO()):
                            pruning_cli()
                    finally:
                        torch.set_num_threads(threads)
                    self.assertTrue(json.loads((args.output/'stage1_pruning/independent_verification.json').read_text())['passed'])
                    log_before = file_sha256(args.output / 'run.log')
                    with self.assertRaises(FileExistsError):
                        entry.main(args)
                    self.assertEqual(log_before, file_sha256(args.output / 'run.log'))
                    if fmt == 'onnx':
                        self.assertEqual(saved['export_format'], 'safetensors')
                        self.assertTrue((args.output / 'stage2_finetuning/recovered_model/model.onnx').is_file())
                    if number == 0:
                        # 相同输入/种子下，比对旧自动两阶段与新显式两阶段的实际权重。
                        old = SimpleNamespace(**vars(args), profile_only=False)
                        old.output = root / 'legacy'
                        with contextlib.redirect_stdout(io.StringIO()):
                            legacy_run(old)
                        original = OTO.load_subnet(old.output / 'lora_recovery/recovered_model')
                        current = OTO.load_subnet(args.output / 'stage2_finetuning/recovered_model')
                        self.assertEqual(original.state_dict().keys(), current.state_dict().keys())
                        for name, tensor in original.state_dict().items():
                            torch.testing.assert_close(current.state_dict()[name], tensor, atol=0, rtol=0)
                        old_result = json.loads((old.output / 'pruning_result.json').read_text())
                        self.assertEqual(result['pruning']['alpha_final'], old_result['alpha_final'])

    def test_disabled_recovery_rejected_and_cli_overrides_recorded(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            args = fixture(Path(temp))
            args.export_format = 'pt'
            args.log_every = 1
            pruning, recovery = entry.load_configuration(args)
            self.assertEqual(pruning.export_format, 'pt')
            self.assertEqual((pruning.log_every, recovery.log_every), (1, 1))
            data = json.loads(args.config.read_text())
            data['recovery']['enabled'] = False
            args.config.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'recovery.enabled=true'):
                entry.main(args)
            self.assertFalse(args.output.exists())

    def test_failure_keeps_completed_stage_and_never_marks_workflow_success(self):
        for failed in ('pruning', 'finetuning'):
            with self.subTest(stage=failed), tempfile.TemporaryDirectory() as temp:
                with contextlib.redirect_stdout(io.StringIO()):
                    args = fixture(Path(temp))
                def saved_pruning(_args, _config, output):
                    output.mkdir()
                    result = {'state': 'completed', 'parameters_before': 100, 'parameters_after': 80,
                              'compressed_package': str(output / 'compressed_model')}
                    (output / 'pruning_result.json').write_text(json.dumps(result))
                    return result
                prune = patch.object(entry, 'prune_model', side_effect=(RuntimeError('pruning_failure')
                                     if failed == 'pruning' else saved_pruning))
                with prune, patch.object(entry, 'run_recovery', side_effect=RuntimeError('finetuning_failure')) as recovery:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                        entry.main(args)
                status = json.loads((args.output / 'status.json').read_text())
                self.assertEqual(status['state'], 'failed')
                self.assertEqual(status['stage'], failed)
                self.assertEqual(status['pruning_result_preserved'], failed == 'finetuning')
                self.assertFalse((args.output / 'workflow_result.json').exists())
                self.assertEqual(recovery.call_count, int(failed == 'finetuning'))

    def test_seven_task_controller_uses_new_entry_and_both_correct_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'config.json'
            config.write_text(json.dumps({'adaptive_alpha': True, 'recovery': {'enabled': True},
                                          'target_group_sparsity': .2}))
            names = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
            paper = {'task_order': names, 'dense_phi2_percent': [80.] * 7}
            output = root / 'experiment'
            commands = []
            class Child:
                pid = 1234
                def __init__(self, command, **kwargs):
                    self.command = command
                    self.stdout = io.StringIO('child progress\n')
                    commands.append(command)
                def wait(self):
                    directory = Path(self.command[self.command.index('--output') + 1])
                    directory.mkdir(parents=True)
                    if directory.name == 'training':
                        target = directory / 'stage1_pruning'
                        target.mkdir()
                        (target / 'pruning_result.json').write_text(json.dumps({'parameter_reduction_ratio': .1}))
                    elif directory.name.startswith('evaluation_'):
                        (directory / 'summary.json').write_text(json.dumps({
                            'full_seven_task_evaluation': True, 'paper_reference': paper,
                            'tasks': {name: {'acc': .5, 'acc_norm': .6} for name in names}}))
                    return 0
            args = SimpleNamespace(source_commit='a' * 40, output=output, data_dir=root / 'data',
                                   config=config, model_dir=root / 'model', physical_gpu=0,
                                   pruning_python='pruning_python', evaluation_python='evaluation_python')
            gpu = {'uuid': 'test-only', 'used_mib': 0, 'util_percent': 0}
            with patch.object(controller, 'load_manifest', return_value={'alpaca_used': False,
                         'source_files': dict.fromkeys(names)}), patch.object(controller, 'file_sha256', return_value='test'), \
                 patch.object(controller, 'gpu_snapshot', return_value=gpu), \
                 patch.object(controller.time, 'sleep'), patch.object(controller.subprocess, 'Popen', Child), \
                 contextlib.redirect_stdout(io.StringIO()):
                controller.run_benchmark_experiment(args)
            self.assertTrue(commands[0][2].endswith('prune_phi2_hesso_cric.py'))
            self.assertTrue(commands[1][2].endswith('pretrain_phi2_pruning.py'))
            self.assertNotIn('--benchmarks',commands[1])
            self.assertIn(str(output / 'training/stage1_pruning/compressed_model'), commands[2])
            self.assertIn(str(output / 'training/stage2_finetuning/recovered_model'), commands[3])
            self.assertEqual(json.loads((output / 'controller_status.json').read_text())['state'], 'completed')
            self.assertIn('child progress', (output / 'training.stdout.log').read_text())

    def test_benchmark_mode_dispatch_and_onnx_preflight(self):
        args=SimpleNamespace(benchmarks=True)
        with patch.object(entry,'run_benchmark_experiment',return_value={'state':'test'}) as runner:
            self.assertEqual(entry.main(args),{'state':'test'})
            runner.assert_called_once_with(args)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'config.json'
            config.write_text(json.dumps({'export_format':'onnx','adaptive_alpha':True,'recovery':{'enabled':True}}))
            args=SimpleNamespace(source_commit='a'*40,output=root/'out',data_dir=root/'data',
                                 config=config,export_format=None,log_every=None)
            names=['boolq','piqa','hellaswag','winogrande','arc_easy','arc_challenge','openbookqa']
            with patch.object(entry,'load_manifest',return_value={'alpaca_used':False,'source_files':dict.fromkeys(names)}):
                with self.assertRaisesRegex(ValueError,'七任务 GPU 评分需要'):
                    entry.run_benchmark_experiment(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
