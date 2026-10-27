"""Phi 本地结构合同测试：异构逐层宽度、实际矩阵缩小、缓存与数据隔离。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import PhiConfig, PhiForCausalLM

from phi2_data import PreparedDataset, collate_examples, load_manifest
from phi2_structure import (apply_structure, build_param_groups, collect_pruning_plan,
                            load_pruned_package, parameter_count, save_pruned_package)
from prepare_phi2_data import prepare
from only_train_once.optimizer import HESSOCRIC


def tiny_model():
    torch.manual_seed(71)
    config = PhiConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                       partial_rotary_factor=.5, resid_pdrop=0., eos_token_id=127, pad_token_id=0)
    config._attn_implementation = 'eager'
    return PhiForCausalLM(config).eval()


def zeroed_model_and_plan():
    model = tiny_model()
    groups = build_param_groups(model)
    with contextlib.redirect_stdout(io.StringIO()):
        optimizer = HESSOCRIC(groups, lr=.01, sampling_steps=2, hybrid_training_steps=4,
                             max_cycle_period=2, adaptive_alpha=False)
    for group in optimizer.param_groups:
        if not group['is_prunable']:
            continue
        # 不同层保留不同宽度，专门检查全局 config 无法独立表达的结构。
        count = group['layer_index'] + 1
        group['pruned_idxes'] = list(range(count))
        optimizer._scale_group(group, group['pruned_idxes'], 0.)
    optimizer.is_cric_terminated = True
    return model, collect_pruning_plan(model, optimizer)


class PhiStructureTests(unittest.TestCase):
    def test_all_parameters_owned_once_and_output_bias_protected(self):
        model = tiny_model()
        groups = build_param_groups(model)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        rest = groups[-1]['p_names']
        self.assertIn('model.layers.0.self_attn.dense.bias', rest)
        self.assertIn('model.layers.0.mlp.fc2.bias', rest)

    def test_physical_ffn_and_attention_match_zeroed_model(self):
        model, plan = zeroed_model_and_plan()
        tokens = torch.arange(8).reshape(1, -1)
        with torch.no_grad():
            reference = model(tokens, use_cache=False).logits
        before = parameter_count(model)
        apply_structure(model, plan)
        with torch.no_grad():
            actual = model(tokens, use_cache=False).logits
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
        self.assertLess(parameter_count(model), before)
        self.assertEqual(model.model.layers[0].mlp.fc1.out_features, 63)
        self.assertEqual(model.model.layers[1].mlp.fc1.out_features, 62)
        self.assertEqual(model.model.layers[0].self_attn.q_proj.out_features, 24)
        self.assertEqual(model.model.layers[1].self_attn.dense.in_features, 16)

    def test_safe_package_reload_backward_and_cached_generation(self):
        model, plan = zeroed_model_and_plan()
        apply_structure(model, plan)
        inputs = torch.arange(8).reshape(1, -1)
        with torch.no_grad():
            expected = model(inputs, use_cache=False).logits
        # 随机微型模型的测试只验证模型结构包；真实 tokenizer 在完整验收中核对。
        tokenizer = SimpleNamespace(save_pretrained=lambda _: None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'package'
            save_pruned_package(model, tokenizer, plan, path)
            other = load_pruned_package(path).eval()
        with torch.no_grad():
            torch.testing.assert_close(other(inputs, use_cache=False).logits, expected)
            generated = other.generate(inputs, max_new_tokens=3, do_sample=False, use_cache=True)
        self.assertGreater(generated.shape[1], inputs.shape[1])
        other.train()
        loss = other(inputs, labels=inputs, use_cache=False).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(other.parameters(), 1., error_if_nonfinite=True)))

    def test_bad_structure_rejected_before_any_shape_change(self):
        model, plan = zeroed_model_and_plan()
        bad = copy.deepcopy(plan)
        bad['layers'][1]['attention']['keep'] = []
        before = [tuple(p.shape) for p in model.parameters()]
        with self.assertRaises(ValueError):
            apply_structure(model, bad)
        self.assertEqual(before, [tuple(p.shape) for p in model.parameters()])

    def test_high_precision_structure_audit_restores_native_float32(self):
        from phi2_precision import audit_structure_precision
        sparse, plan = zeroed_model_and_plan()
        compact = copy.deepcopy(sparse)
        apply_structure(compact, plan)
        audit, _ = audit_structure_precision(sparse, compact, {'input_ids': torch.arange(8).reshape(1, -1)})
        self.assertTrue(audit['structure_fp64']['passed'])
        self.assertEqual(next(sparse.parameters()).dtype, torch.float32)
        self.assertEqual(next(compact.parameters()).dtype, torch.float32)

    def test_nonzero_pruned_rows_rejected(self):
        model, plan = zeroed_model_and_plan()
        groups = build_param_groups(model)
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = HESSOCRIC(groups, lr=.01, sampling_steps=2, hybrid_training_steps=4)
        optimizer.is_cric_terminated = True
        optimizer.param_groups[0]['pruned_idxes'] = [1]  # 第 1 行未被清零。
        with self.assertRaisesRegex(RuntimeError, '没有精确归零'):
            collect_pruning_plan(model, optimizer)

    def test_preparation_split_integrity_and_padding(self):
        class Tokenizer:
            eos_token_id = 127
            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 100 for c in text]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / 'source.json'
            source.write_text(json.dumps([{'instruction': str(i), 'input': '', 'output': 'answer'}
                                          for i in range(12)]), encoding='utf-8')
            args = SimpleNamespace(source=source, model_dir=tmp, output=tmp/'prepared', seed=42,
                                   max_length=32, train_samples=4, alpha_samples=2, eval_samples=2)
            with patch('prepare_phi2_data.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(args)
            manifest = load_manifest(args.output)
            data = PreparedDataset(args.output, 'train', manifest)
            batch = collate_examples([data[0], {'row_id': -1, 'input_ids': [1, 2], 'labels': [-100, 2]}], 127)
            self.assertTrue(torch.all(batch['labels'][batch['attention_mask'] == 0] == -100))
            self.assertEqual(data[0]['labels'][-1], 127)
            file = args.output / 'alpha.jsonl'
            file.write_text(file.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '摘要'):
                load_manifest(args.output)

    def test_refuse_existing_run_without_overwriting_its_failure_record(self):
        from prune_phi2_hesso_cric import main
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / 'failure.json'
            sentinel.write_text('preserve existing record', encoding='utf-8')
            with patch('sys.argv', ['prune', '--data-dir', tmp, '--output', tmp]):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserve existing record')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
