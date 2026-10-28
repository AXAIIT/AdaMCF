"""Phi 本地结构合同测试：异构逐层宽度、实际矩阵缩小、缓存与数据隔离。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import PhiConfig, PhiForCausalLM

from phi2_data import PreparedDataset, collate_examples, load_manifest
from only_train_once.model_adapters.phi import (apply_structure, build_param_groups, collect_pruning_plan,
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
    def test_preparation_filters_empty_responses_without_rewriting_raw_data(self):
        class Tokenizer:
            eos_token_id=127
            def encode(self,text,add_special_tokens=False):return [ord(c)%100 for c in text]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'source.json'
            source.write_text(json.dumps([{'instruction':'empty','input':'','output':'  '}]
                +[{'instruction':str(i),'input':'','output':'answer'} for i in range(6)]),encoding='utf-8')
            before=source.read_bytes()
            args=SimpleNamespace(source=source,model_dir=tmp,output=tmp/'prepared',seed=42,
                                 max_length=32,train_samples=2,alpha_samples=2,eval_samples=2)
            with patch('prepare_phi2_data.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                with contextlib.redirect_stdout(io.StringIO()):prepare(args)
            manifest=load_manifest(args.output)
            self.assertEqual(manifest['filtered_empty_response_rows'],1)
            self.assertEqual(manifest['eligible_rows'],6)
            self.assertEqual(source.read_bytes(),before)
            self.assertFalse(any(0 in x['row_ids'] for x in manifest['splits'].values()))
            self.assertTrue((args.output/'prepare.log').is_file())

    def test_modes_have_exclusive_candidates_with_complete_parameter_ownership(self):
        for mode, expected in [('ffn', 'ffn'), ('num_head', 'attention'), ('head_dim', 'head_dim')]:
            with self.subTest(mode=mode):
                model = tiny_model()
                groups = build_param_groups(model, mode)
                candidates = [g for g in groups if g['is_prunable']]
                self.assertEqual({g['phi_kind'] for g in candidates}, {expected})
                self.assertEqual(len(candidates), 2)
                ids = [id(p) for g in groups for p in g['params']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids), {id(p) for p in model.parameters()})

    def test_real_cric_selects_each_mode_and_exports_correct_structure(self):
        from phi2_precision import audit_structure_precision
        for mode in ['ffn', 'num_head', 'head_dim']:
            with self.subTest(mode=mode):
                model = tiny_model()
                groups = build_param_groups(model, mode)
                with contextlib.redirect_stdout(io.StringIO()):
                    optimizer = HESSOCRIC(groups, lr=.001, first_momentum=0., target_group_sparsity=.25,
                                         start_cric_step=1, sampling_steps=2, hybrid_training_steps=3,
                                         max_cycle_period=2, alpha_scheduler_cfg={'warmup_updates':0, 'min_gap_std':1e9})
                    optimizer.phi_pruning_mode = mode
                    tokens = torch.arange(8).reshape(1, -1)
                    for _ in range(32):
                        optimizer.zero_grad()
                        loss = model(tokens, labels=tokens, use_cache=False).loss
                        loss.backward()
                        if optimizer.will_update_alpha_on_next_step():
                            optimizer.alpha_scheduler.update(model, None, [({'input_ids':tokens,'labels':tokens},None)])
                        optimizer.step(loss.detach())
                plan = collect_pruning_plan(model, optimizer)
                self.assertGreater(optimizer.alpha_scheduler.update_count, 0)
                if mode == 'ffn':
                    self.assertGreater(sum(len(e['ffn']['pruned']) for e in plan['layers']), 0)
                    self.assertEqual(sum(len(e['attention']['pruned']) for e in plan['layers']), 0)
                elif mode == 'num_head':
                    self.assertGreater(sum(len(e['attention']['pruned']) for e in plan['layers']), 0)
                    self.assertEqual(sum(len(e['ffn']['pruned']) for e in plan['layers']), 0)
                else:
                    self.assertGreater(sum(len(e['head_dim_pruned_units']) for e in plan['layers']), 0)
                    self.assertEqual(sum(len(e['ffn']['pruned']) for e in plan['layers']), 0)
                    self.assertEqual(sum(len(e['attention']['pruned']) for e in plan['layers']), 0)
                compact = copy.deepcopy(model)
                apply_structure(compact, plan)
                audit, _ = audit_structure_precision(model, compact, {'input_ids': tokens})
                self.assertTrue(audit['structure_fp64']['passed'])
                self.assertLess(parameter_count(compact), parameter_count(model))
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp) / 'package'
                    save_pruned_package(compact, SimpleNamespace(save_pretrained=lambda _:None), plan, folder)
                    other = load_pruned_package(folder).eval()
                with torch.no_grad():
                    torch.testing.assert_close(other(tokens, use_cache=False).logits,
                                               compact(tokens, use_cache=False).logits)
                    cached = other(tokens, use_cache=True)
                    for i, entry in enumerate(plan['layers']):
                        key, value = cached.past_key_values[i]
                        self.assertEqual(key.shape[1], len(entry['attention']['keep']))
                        self.assertEqual(key.shape[-1], len(entry['head_dim_keep']))
                    next_token = cached.logits[:, -1:].argmax(-1)
                    continuation = other(next_token, past_key_values=cached.past_key_values, use_cache=True).logits
                    full = other(torch.cat([tokens, next_token], dim=1), use_cache=False).logits[:, -1:]
                    torch.testing.assert_close(continuation, full, atol=1e-6, rtol=1e-5)
                other.train()
                other(tokens, labels=tokens, use_cache=False).loss.backward()
                self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(other.parameters(),1.,error_if_nonfinite=True)))

    def test_head_dim_rejects_broken_rope_pair(self):
        model, plan = zeroed_model_and_plan()
        entry = plan['layers'][0]
        entry['head_dim_pruned'] = [0]
        entry['head_dim_keep'] = list(range(1, 8))
        with self.assertRaisesRegex(ValueError, '配对'):
            apply_structure(model, plan)

    def test_head_dim_can_remove_entire_rotary_region_without_rebasing(self):
        from phi2_precision import audit_structure_precision
        model = tiny_model()
        groups = build_param_groups(model, 'head_dim')
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = HESSOCRIC(groups, lr=.01, sampling_steps=2, hybrid_training_steps=3)
        optimizer.phi_pruning_mode = 'head_dim'
        for group in optimizer.param_groups:
            if group['is_prunable']:
                group['pruned_idxes'] = [0,1]
                optimizer._scale_group(group, [0,1], 0.)
        optimizer.is_cric_terminated = True
        plan = collect_pruning_plan(model, optimizer)
        compact = copy.deepcopy(model)
        apply_structure(compact, plan)
        self.assertTrue(all(layer.self_attn.rotary_ndims == 0 for layer in compact.model.layers))
        audit, _ = audit_structure_precision(model, compact, {'input_ids':torch.arange(8).reshape(1,-1)})
        self.assertTrue(audit['structure_fp64']['passed'])

    def test_previous_version1_structure_package_remains_loadable(self):
        model, plan = zeroed_model_and_plan()
        old = {'format_version':1, 'layers':[
            {key: entry[key] for key in ('layer_index','ffn','attention','head_dim')}
            for entry in plan['layers']]}
        apply_structure(model, old)
        tokens = torch.arange(8).reshape(1,-1)
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'old_package'
            save_pruned_package(model, SimpleNamespace(save_pretrained=lambda _:None), old, package)
            # 旧运行包没有序列化字段，验证 loader 的真实向后兼容路径。
            architecture=json.loads((package/'architecture.json').read_text())
            architecture.pop('weights_format',None);architecture.pop('weights_file',None)
            (package/'architecture.json').write_text(json.dumps(architecture),encoding='utf-8')
            restored = load_pruned_package(package).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(tokens, use_cache=False).logits,
                                       model(tokens, use_cache=False).logits)

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
