"""框架集成回归：统一 OTO 入口、真实 CRIC、小模型与旧算子共存。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from test_phi2_contract import tiny_model, zeroed_model_and_plan
from phi2_precision import audit_structure_precision
from only_train_once.model_adapters.phi import load_pruned_package
from only_train_once import OTO
from only_train_once.model_adapters import create_model_adapter
from only_train_once.operation.operator import COMPOSED_MODULES, PhiAttentionOTO


class PhiFrameworkTests(unittest.TestCase):
    def test_export_only_cli_reuses_sparse_checkpoint_and_refuses_overwrite(self):
        from prune_phi2_hesso_cric import main
        from verify_phi2_pruning import verify
        model, plan = zeroed_model_and_plan()
        tokens = torch.arange(8).reshape(1,-1)
        batch = {'input_ids':tokens, 'labels':tokens}
        with torch.no_grad():reference = model(**batch, use_cache=False).logits
        for fmt in ('pt','safetensors','onnx'):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                model.save_pretrained(run/'sparse_model',safe_serialization=True)
                (run/'pruning_plan.json').write_text(json.dumps(plan),encoding='utf-8')
                (run/'export_context.json').write_text(json.dumps({
                    'parameters_before':sum(p.numel() for p in model.parameters())}),encoding='utf-8')
                torch.save({'batch':batch,'sparse_logits':reference,'atol':1e-6,'rtol':1e-5},run/'sparse_probe.pt')
                argv=['prune','--export-only','--output',str(run),'--device','cpu','--export-format',fmt]
                tokenizer=SimpleNamespace(save_pretrained=lambda _:None)
                with patch('sys.argv',argv), patch('prune_phi2_hesso_cric.AutoTokenizer.from_pretrained',return_value=tokenizer):
                    with contextlib.redirect_stdout(io.StringIO()):main()
                loaded = OTO.load_subnet(run/'compressed_model')
                probe = torch.load(run/'equivalence_probe.pt',weights_only=True)
                if fmt=='onnx':
                    torch.testing.assert_close(torch.from_numpy(loaded(tokens)),probe['compressed_logits'],atol=1e-5,rtol=1e-4)
                    verify(SimpleNamespace(run_dir=run,device='cpu'))
                    self.assertTrue(json.loads((run/'independent_verification.json').read_text())['inference_only'])
                else:
                    loaded.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(loaded(**batch,use_cache=False).logits,probe['compressed_logits'])
                events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['stage']=='完成' for e in events))
                self.assertTrue(all(e['time'].endswith('+08:00') for e in events))
                self.assertIn('6/6 导出',(run/'run.log').read_text())
                result = (run/'pruning_result.json').read_bytes()
                with patch('sys.argv',argv), self.assertRaises(FileExistsError):main()
                self.assertEqual((run/'pruning_result.json').read_bytes(),result)

    def test_training_cli_requires_dataset_without_creating_output(self):
        from prune_phi2_hesso_cric import main
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new_run'
            with patch('sys.argv',['prune','--output',str(output)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):main()
            self.assertFalse(output.exists())

    def test_cric_loop_writes_readable_log_and_structured_step_trace(self):
        from prune_phi2_hesso_cric import train_cric,PruningConfig
        model=tiny_model();owner=OTO(model,pruning_mode='ffn')
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer=owner.hessocric(lr=.001,first_momentum=0.,target_group_sparsity=.25,
                start_cric_step=1,sampling_steps=2,max_cycle_period=2,hybrid_training_steps=3,
                alpha_scheduler_cfg={'warmup_updates':0,'min_gap_std':1e9})
        config=PruningConfig(warmup_steps=0,sampling_steps=2,max_cycle_period=2,
                             hybrid_training_steps=3,post_pruning_steps=1,log_every=10)
        tokens=torch.arange(8).reshape(1,-1);alpha=tokens+8
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                final=train_cric(model,optimizer,[{'input_ids':tokens,'labels':tokens}],
                    [{'input_ids':alpha,'labels':alpha}],config,output,torch.device('cpu'))
            trace=[json.loads(line) for line in (output/'training_trace.jsonl').read_text().splitlines()]
            self.assertEqual(final['phase'],'post_pruning')
            self.assertEqual(len(trace),final['step'])
            self.assertTrue(all('learning_rate' in row and 'pruned_logical_groups' in row for row in trace))
            self.assertIn('CRIC 进度',(output/'run.log').read_text())
            self.assertTrue((output/'events.jsonl').is_file())

    def test_auto_dispatch_uses_native_adapter_without_onnx(self):
        with patch('only_train_once.Graph', side_effect=AssertionError('不应调用追踪器')):
            for mode in ('joint', 'ffn', 'num_head', 'head_dim'):
                oto = OTO(tiny_model(), pruning_mode=mode)
                self.assertEqual(oto.model_adapter_name, 'native_phi')
                self.assertIsNone(oto._graph)
                self.assertEqual(oto.compute_num_params(in_million=False),
                                 sum(p.numel() for p in oto._model.parameters()))

    def test_real_cric_uses_factory_and_implicit_export_for_three_modes(self):
        for mode in ('ffn', 'num_head', 'head_dim'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                model = tiny_model()
                oto = OTO(model, pruning_mode=mode)
                with contextlib.redirect_stdout(io.StringIO()):
                    optimizer = oto.hessocric(lr=.001, first_momentum=0.,
                        target_group_sparsity=.25, start_cric_step=1, sampling_steps=2,
                        hybrid_training_steps=3, max_cycle_period=2,
                        alpha_scheduler_cfg={'warmup_updates':0, 'min_gap_std':1e9})
                    tokens = torch.arange(8).reshape(1,-1)
                    for _ in range(32):
                        optimizer.zero_grad()
                        loss = model(tokens, labels=tokens, use_cache=False).loss
                        loss.backward()
                        if optimizer.will_update_alpha_on_next_step():
                            optimizer.alpha_scheduler.update(model, None,
                                [({'input_ids':tokens, 'labels':tokens}, None)])
                        optimizer.step(loss.detach())
                plan = oto.collect_pruning_plan()
                kind = {'ffn':'ffn','num_head':'attention','head_dim':'head_dim'}[mode]
                removed = sum(len(e['head_dim_pruned']) if kind == 'head_dim'
                              else len(e[kind]['pruned']) for e in plan['layers'])
                self.assertGreater(removed, 0)
                self.assertGreater(optimizer.alpha_scheduler.update_count, 0)
                sparse = copy.deepcopy(model)
                oto.construct_subnet(out_dir=tmp, ckpt_format='safetensors')
                self.assertEqual(oto.pruning_plan['pruning_mode'], mode)
                self.assertTrue(Path(oto.full_group_sparse_model_path).is_dir())
                audit, _ = audit_structure_precision(sparse, model, {'input_ids':tokens})
                self.assertTrue(audit['structure_fp64']['passed'])
                restored = OTO.load_subnet(oto.compressed_model_path).eval()
                old_loader = load_pruned_package(oto.compressed_model_path).eval()
                with torch.no_grad():
                    ref = model(tokens, use_cache=False).logits
                    torch.testing.assert_close(restored(tokens, use_cache=False).logits, ref)
                    torch.testing.assert_close(old_loader(tokens, use_cache=False).logits, ref)
                    cached = restored(tokens, use_cache=True)
                    for i, entry in enumerate(plan['layers']):
                        key, value = cached.past_key_values[i]
                        self.assertEqual(key.shape, value.shape)
                        self.assertEqual(key.shape[1], len(entry['attention']['keep']))
                        self.assertEqual(key.shape[-1], len(entry['head_dim_keep']))
                    nxt = cached.logits[:,-1:].argmax(-1)
                    continued = restored(nxt, past_key_values=cached.past_key_values, use_cache=True)
                    full = restored(torch.cat((tokens,nxt),dim=1), use_cache=False).logits[:,-1:]
                    torch.testing.assert_close(continued.logits, full, atol=1e-6, rtol=1e-5)
                    self.assertGreater(restored.generate(tokens,max_new_tokens=2,use_cache=True).shape[1],8)
                restored.train()
                restored(tokens, labels=tokens, use_cache=False).loss.backward()
                self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(restored.parameters(),1.)))
                with self.assertRaisesRegex(RuntimeError, '重复'):
                    oto.construct_subnet(out_dir=Path(tmp)/'again')

    def test_external_plan_is_checked_before_any_mutation(self):
        model, plan = zeroed_model_and_plan()
        model.model.layers[1].mlp.fc1.bias.data[0] = 1.
        before = [tuple(p.shape) for p in model.parameters()]
        oto = OTO(model)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, '没有精确归零'):
                oto.construct_subnet(out_dir=tmp, pruning_plan=plan)
            self.assertFalse((Path(tmp)/'compressed_model').exists())
        self.assertEqual(before, [tuple(p.shape) for p in model.parameters()])

    def test_incomplete_cric_and_wrong_optimizer_rejected(self):
        oto = OTO(tiny_model())
        with self.assertRaisesRegex(RuntimeError, '先调用'):
            oto.collect_pruning_plan()
        with contextlib.redirect_stdout(io.StringIO()):
            oto.hessocric(lr=.01,sampling_steps=2,hybrid_training_steps=3)
        with self.assertRaisesRegex(RuntimeError, 'CRIC 尚未结束'):
            oto.collect_pruning_plan()
        other = OTO(tiny_model())
        with contextlib.redirect_stdout(io.StringIO()):
            opt = other.hessocric(lr=.01,sampling_steps=2,hybrid_training_steps=3)
        with self.assertRaisesRegex(ValueError,'不属于'):
            oto._model_adapter.collect_pruning_plan(opt)

    def test_existing_package_and_unsupported_options_rejected(self):
        model, plan = zeroed_model_and_plan()
        oto = OTO(model)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'compressed_model').mkdir()
            for kwargs, error in [({},FileExistsError),({'ckpt_format':'invalid'},ValueError),
                                  ({'merge_lora_to_base':True},ValueError)]:
                with self.assertRaises(error):
                    oto.construct_subnet(out_dir=tmp, pruning_plan=plan, **kwargs)
        self.assertFalse(oto._model_adapter.exported)

    def test_three_explicit_formats_export_load_and_preserve_dynamic_logits(self):
        for mode in ('joint','head_dim'):
            if mode == 'joint':
                sparse,plan = zeroed_model_and_plan()
            else:
                sparse = tiny_model();owner = OTO(sparse,pruning_mode=mode)
                with contextlib.redirect_stdout(io.StringIO()):
                    optimizer = owner.hessocric(lr=.01,sampling_steps=2,hybrid_training_steps=3,adaptive_alpha=False)
                for group in optimizer.param_groups:
                    if group['is_prunable']:
                        group['pruned_idxes']=list(range(group['layer_index']+1))
                        optimizer._scale_group(group,group['pruned_idxes'],0.)
                optimizer.is_cric_terminated=True
                plan=owner.collect_pruning_plan()
            for fmt,name in [('pt','model.pt'),('safetensors','model.safetensors'),('onnx','model.onnx')]:
                with self.subTest(mode=mode,format=fmt),tempfile.TemporaryDirectory() as tmp:
                    compact=copy.deepcopy(sparse);oto=OTO(compact,pruning_mode=mode)
                    oto.construct_subnet(out_dir=tmp,pruning_plan=plan,ckpt_format=fmt,save_full_group_sparse_model=False)
                    package=Path(oto.compressed_model_path)
                    self.assertTrue((package/name).is_file())
                    for other in {'model.pt','model.safetensors','model.onnx'}-{name}:
                        self.assertFalse((package/other).exists())
                    loaded=OTO.load_subnet(package)
                    for batch,length in ((1,3),(2,7),(1,11)):
                        ids=torch.arange(batch*length).reshape(batch,length)%128
                        mask=torch.ones_like(ids)
                        if batch==2:mask[1,-2:]=0
                        with torch.no_grad():reference=compact(ids,attention_mask=mask,use_cache=False).logits
                        if fmt=='onnx':
                            actual=torch.from_numpy(loaded(ids,mask))
                        else:
                            loaded.eval()
                            with torch.no_grad():actual=loaded(ids,attention_mask=mask,use_cache=False).logits
                        torch.testing.assert_close(actual,reference,atol=1e-5,rtol=1e-4)
                    if fmt=='onnx':
                        with self.assertRaisesRegex(RuntimeError,'推理格式'):loaded.train()

    def test_lora_is_not_accepted_as_a_pruning_candidate(self):
        model=tiny_model()
        model.register_parameter('lora_A',nn.Parameter(torch.ones(2,32)))
        with self.assertRaisesRegex(ValueError,'先剪枝'):
            OTO(model)

    def test_optional_float16_package_preserves_dtype(self):
        model, plan = zeroed_model_and_plan()
        oto = OTO(model)
        with tempfile.TemporaryDirectory() as tmp:
            oto.construct_subnet(out_dir=tmp, pruning_plan=plan, export_float16=True,
                                 save_full_group_sparse_model=False)
            loaded = OTO.load_subnet(oto.compressed_model_path)
            self.assertEqual(next(loaded.parameters()).dtype, torch.float16)

    def test_legacy_phi_registration_and_non_phi_dispatch_unchanged(self):
        self.assertIs(COMPOSED_MODULES['PhiMHA'], PhiAttentionOTO)
        self.assertNotIn('PhiAttention', COMPOSED_MODULES)
        class PhiMHA(nn.Module):
            pass
        self.assertIsNone(create_model_adapter(PhiMHA()))
        self.assertIsNone(create_model_adapter(nn.Linear(4,4)))
        self.assertIsNone(create_model_adapter(tiny_model(), 'graph'))
        with self.assertRaises(TypeError):
            create_model_adapter(nn.Linear(4,4), 'native_phi')

    def test_legacy_graph_construct_subnet_still_executes_real_compression(self):
        torch.manual_seed(9)
        model = nn.Sequential(nn.Linear(4,6),nn.ReLU(),nn.Linear(6,2)).eval()
        x = torch.randn(2,4)
        oto = OTO(model, x)
        self.assertEqual(oto.model_adapter_name, 'graph')
        with torch.no_grad():
            model[0].weight[0].zero_(); model[0].bias[0].zero_(); model[2].weight[:,0].zero_()
            reference = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            oto.construct_subnet(out_dir=tmp)
            restored = torch.load(oto.compressed_model_path, weights_only=False)
        self.assertEqual(model[0].out_features, 5)
        self.assertEqual(model[2].in_features, 5)
        torch.testing.assert_close(restored(x), reference)

    def test_legacy_cnn_graph_compression_preserves_forward(self):
        model = nn.Sequential(nn.Conv2d(3,6,1),nn.ReLU(),nn.Conv2d(6,4,1)).eval()
        x = torch.randn(1,3,8,8)
        oto = OTO(model,x)
        self.assertEqual(oto.model_adapter_name,'graph')
        with torch.no_grad():
            model[0].weight[0].zero_();model[0].bias[0].zero_();model[2].weight[:,0].zero_()
            ref = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            oto.construct_subnet(out_dir=tmp)
            loaded = torch.load(oto.compressed_model_path,weights_only=False)
        self.assertEqual(model[0].out_channels,5)
        self.assertEqual(model[2].in_channels,5)
        torch.testing.assert_close(loaded(x),ref)

    def test_native_protection_and_graph_only_interfaces_are_explicit(self):
        oto = OTO(tiny_model())
        oto.mark_unprunable_by_param_names(['model.layers.0.self_attn.q_proj.weight'])
        groups = {g['id']:g for g in oto.get_param_groups()}
        self.assertFalse(groups['layer_00_attention']['is_prunable'])
        self.assertTrue(groups['layer_01_attention']['is_prunable'])
        with self.assertRaises(NotImplementedError):oto.compute_flops()
        with self.assertRaises(NotImplementedError):oto.partition_pzigs()
        with self.assertRaises(ValueError):OTO(tiny_model(),model_adapter='invalid')
        with self.assertRaises(ValueError):OTO(tiny_model(),onnx_opset_version=17)

    def test_reloaded_package_inference_allowed_but_repruning_is_explicit(self):
        model,plan = zeroed_model_and_plan()
        oto = OTO(model)
        with tempfile.TemporaryDirectory() as tmp:
            oto.construct_subnet(out_dir=tmp,pruning_plan=plan,save_full_group_sparse_model=False)
            loaded = OTO.load_subnet(oto.compressed_model_path)
        self.assertTrue(torch.isfinite(loaded(torch.arange(8).reshape(1,-1)).logits).all())
        with self.assertRaisesRegex(ValueError,'再次结构剪枝'):
            OTO(loaded)

    def test_importing_framework_does_not_require_transformers(self):
        code = """import builtins
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name == 'transformers' or name.startswith('transformers.'):
        raise ImportError('blocked optional dependency')
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from only_train_once import OTO
from torch import nn
from only_train_once.model_adapters import create_model_adapter
assert create_model_adapter(nn.Linear(2,2)) is None
print('OPTIONAL_DEPENDENCY_PASSED')
"""
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
