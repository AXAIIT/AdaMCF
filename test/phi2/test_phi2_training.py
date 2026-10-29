"""两阶段回归：真实小型 Phi 剪枝/LoRA、α 取样与三格式，不下载模型。"""
from dataclasses import asdict
import contextlib
import copy
import io
import json
import os
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
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from test_phi2_contract import tiny_model
from phi2_data import collate_alpha_examples
from only_train_once.optimizer.alpha_data import AlphaBatch
from finetune_phi2_lora import RecoveryConfig, attach_lora, finalize_saved_recovery, load_recovery, run_recovery, validate_config
from phi2_precision import merge_lora_with_precision_audit
from phi2_data import file_sha256
from test_phi2_contract import prepare
from prune_phi2_hesso_cric import PruningConfig, run, train_cric
from only_train_once import OTO


def tokenizer():
    vocab={'[PAD]':0,'[UNK]':1,**{f'w{i}':i+2 for i in range(125)},'[EOS]':127}
    engine=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));engine.pre_tokenizer=Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=engine,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')


def alpha_loader(seed=87):
    rows=[{'row_id':i,'input_ids':[i+2,i+3], 'labels':[-100,i+3]} for i in range(6)]
    return DataLoader(rows,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(seed),
                      collate_fn=lambda rows:collate_alpha_examples(rows,0))


class PhiTrainingTests(unittest.TestCase):
    def test_alpha_collator_only_provides_causal_lm_data_and_metadata(self):
        item=next(iter(alpha_loader()))
        self.assertIsInstance(item,AlphaBatch)
        self.assertIsNone(item.targets)
        self.assertEqual(len(item.sample_ids),1)
        self.assertEqual(item.supervised_tokens,1)
        self.assertEqual(item.inputs['input_ids'].shape,(1,2))

    def test_real_cric_trace_records_alpha_row_ids_and_supervised_tokens(self):
        model=tiny_model();owner=OTO(model,pruning_mode='ffn')
        with contextlib.redirect_stdout(io.StringIO()):
            opt=owner.hessocric(lr=.001,first_momentum=0,target_group_sparsity=.25,start_cric_step=1,
                sampling_steps=2,max_cycle_period=3,hybrid_training_steps=2,
                alpha_scheduler_cfg={'warmup_updates':0,'min_gap_std':1e9,'eval_batch_limit':2})
        config=PruningConfig(warmup_steps=0,sampling_steps=2,max_cycle_period=3,
                             hybrid_training_steps=2,post_pruning_steps=0,log_every=100)
        tokens=torch.arange(6).reshape(1,-1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            train_cric(model,opt,[{'input_ids':tokens,'labels':tokens}],alpha_loader(),
                       config,Path(tmp),torch.device('cpu'))
            trace=[json.loads(line) for line in (Path(tmp)/'training_trace.jsonl').read_text().splitlines()]
        updates=[row for row in trace if row['alpha_updated']]
        self.assertGreaterEqual(len(updates),2)
        self.assertGreater(updates[-1]['alpha_unique_samples_seen'],1)
        self.assertTrue(all(row['alpha_row_ids'] and row['alpha_supervised_tokens']>0 for row in updates))
        self.assertTrue(all(not row['alpha_row_ids'] for row in trace if not row['alpha_updated']))

    def test_lora_only_ab_trainable_on_actual_projection_shapes(self):
        base=tiny_model()
        model,trainable=attach_lora(base,RecoveryConfig(rank=2,dropout=0))
        self.assertTrue(trainable)
        self.assertTrue(all('lora_A' in name or 'lora_B' in name for name,p in trainable))
        self.assertEqual({id(p) for _,p in trainable},{id(p) for p in model.parameters() if p.requires_grad})

    def test_merge_audit_checks_nonzero_adapters_and_rejects_corruption(self):
        def adapted():
            model,trainable=attach_lora(tiny_model(),RecoveryConfig(rank=2,dropout=0))
            with torch.no_grad():
                for name,p in trainable:
                    if 'lora_B' in name:p.normal_(std=.01)
            return model.eval()
        ids=torch.arange(6).reshape(1,-1)
        probes=[{'input_ids':ids,'attention_mask':torch.ones_like(ids)}]
        model=adapted()
        merged,audit,outputs=merge_lora_with_precision_audit(model,probes)
        self.assertTrue(audit['merge_fp64']['passed'])
        self.assertTrue(all(p.dtype==torch.float32 for p in merged.parameters()))
        self.assertEqual(outputs[0].shape[:2],ids.shape)
        model=adapted();kind=type(model.base_model);original=kind.merge_and_unload
        def corrupt(owner,*args,**kwargs):
            result=original(owner,*args,**kwargs)
            with torch.no_grad():result.model.layers[0].mlp.fc2.weight[0,0]+=.1
            return result
        with patch.object(kind,'merge_and_unload',corrupt),self.assertRaises(AssertionError):
            merge_lora_with_precision_audit(model,probes)

    def test_invalid_lora_recipe_and_onnx_base_fail_before_creating_output(self):
        for overrides in ({'rank':0},{'steps':0},{'target_modules':['lm_head']},{'dropout':1.},
                          {'learning_rate':float('nan')},{'warmup_steps':100}):
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):
                validate_config(RecoveryConfig(**overrides))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'base').mkdir()
            (root/'base/architecture.json').write_text(json.dumps({'weights_format':'onnx'}))
            with self.assertRaisesRegex(ValueError,'ONNX 基座不能训练'):
                run_recovery(root/'base',root/'absent',root/'out',RecoveryConfig(), 'cpu')
            self.assertFalse((root/'out').exists())

    def test_two_stage_pipeline_all_pruning_modes_and_export_formats(self):
        # joint 覆盖三格式；另两项确保整头/头内维度变化后，LoRA 使用实际缩小的矩阵。
        cases=[('joint','pt'),('joint','safetensors'),('joint','onnx'),('num_head','pt'),('head_dim','safetensors'),('ffn','pt')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'original';base.mkdir()
            tiny_model().save_pretrained(base);tokenizer().save_pretrained(base)
            source=root/'source.json'
            source.write_text(json.dumps([{'instruction':f'w{i}','input':'w2','output':'w3 w4 w5'} for i in range(20)]))
            data=root/'data'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(source=source,model_dir=base,output=data,seed=42,max_length=32,
                                        train_samples=10,alpha_samples=6,eval_samples=4))
            data_hash=file_sha256(data/'manifest.json')
            for mode,fmt in cases:
                with self.subTest(mode=mode,format=fmt):
                    output=root/f'{mode}_{fmt}'
                    cfg=PruningConfig(pruning_mode=mode,export_format=fmt,warmup_steps=0,
                                      target_group_sparsity=.25,sampling_steps=2,max_cycle_period=2,
                                      hybrid_training_steps=3,post_pruning_steps=0,alpha_eval_batches=2,
                                      log_every=100,recovery=asdict(RecoveryConfig(enabled=True,steps=2,
                                      gradient_accumulation_steps=2,rank=2,dropout=0,eval_every=1,log_every=1)))
                    config=root/f'{mode}_{fmt}.json';config.write_text(json.dumps(asdict(cfg)))
                    with contextlib.redirect_stdout(io.StringIO()):
                        run(SimpleNamespace(config=config,model_dir=base,data_dir=data,output=output,
                                            device='cpu',profile_only=False,export_format=None))
                    result=json.loads((output/'workflow_result.json').read_text())
                    self.assertTrue(result['two_stage']);recovery=result['recovery']
                    self.assertTrue(recovery['base_parameters_unchanged'])
                    self.assertTrue(recovery['adapter_parameters_changed'])
                    self.assertEqual(recovery['export_format'],fmt)
                    self.assertFalse(result['full_benchmark_evaluated'])
                    self.assertEqual(result['pruning']['final_training_step']['phase'],'hybrid')
                    self.assertEqual(file_sha256(data/'manifest.json'),data_hash)
                    probe=torch.load(output/'lora_recovery/recovery_probe.pt',weights_only=True)
                    restored=load_recovery(output/'lora_recovery')
                    with torch.no_grad():actual=restored(**probe['batch'],use_cache=False).logits
                    torch.testing.assert_close(actual,probe['logits'],atol=2e-4,rtol=2e-4)
                    bindings=json.loads((output/'lora_recovery/recovery_manifest.json').read_text())
                    self.assertEqual(bindings['base_files'],{p.name:file_sha256(p) for p in (output/'compressed_model').iterdir() if p.is_file()})
                    self.assertTrue(recovery['merge_precision_audit']['merge_fp64']['passed'])
                    if mode=='ffn':
                        # 导出失败后复用已保存训练结果，不创建优化器、不重训、不改 adapter。
                        retry=output/'saved_adapter_retry'
                        shutil.copytree(output/'lora_recovery',retry,ignore=shutil.ignore_patterns(
                            'recovered_model','recovery_result.json','recovery_probe.pt','merge_precision_audit.json'))
                        adapter_hash=file_sha256(retry/'adapter/adapter_model.safetensors')
                        with patch('torch.optim.AdamW',side_effect=AssertionError('禁止重训')):
                            finalized=finalize_saved_recovery(retry,'pt')
                        self.assertTrue(finalized['finalized_from_saved_adapter'])
                        self.assertEqual(file_sha256(retry/'adapter/adapter_model.safetensors'),adapter_hash)
                        with self.assertRaises(FileExistsError):finalize_saved_recovery(retry,'pt')
                        # 缺少新式 training_result 时，旧完整轨迹也能恢复；不完整轨迹必须拒绝。
                        legacy=output/'legacy_saved_adapter'
                        shutil.copytree(retry,legacy,ignore=shutil.ignore_patterns(
                            'recovered_model','recovery_result.json','recovery_probe.pt','merge_precision_audit.json','training_result.json'))
                        with patch('torch.optim.AdamW',side_effect=AssertionError('禁止重训')):
                            finalized=finalize_saved_recovery(legacy,'pt')
                        self.assertEqual(finalized['training_metadata_source'],'legacy_trace_and_events')
                        broken=output/'incomplete_adapter'
                        shutil.copytree(legacy,broken,ignore=shutil.ignore_patterns('recovered_model','recovery_result.json'))
                        (broken/'finetuning_trace.jsonl').write_text('',encoding='utf-8')
                        with self.assertRaisesRegex(ValueError,'训练不完整'):finalize_saved_recovery(broken,'pt')
                        self.assertFalse((broken/'recovered_model').exists())
                    # 若指定 ONNX，阶段一仍保留可训练中间包，最终恢复模型输出 ONNX。
                    if fmt=='onnx':
                        self.assertTrue((output/'compressed_model/model.safetensors').is_file())
                        self.assertTrue((output/'lora_recovery/recovered_model/model.onnx').is_file())
                    with self.assertRaises(FileExistsError):
                        run_recovery(output/'compressed_model',data,output/'lora_recovery',RecoveryConfig(),'cpu')
                    with self.assertRaisesRegex(ValueError,'基座摘要不一致'):
                        load_recovery(output/'lora_recovery',base_dir=base)
                    evidence=os.environ.get('PHI2_TEST_EVIDENCE_DIR')
                    if evidence:
                        destination=Path(evidence)/f'{mode}_{fmt}'
                        destination.mkdir(parents=True,exist_ok=True)
                        # 保存可审查日志/结果；小模型和临时样本仍由临时目录回收。
                        for relative in ('workflow_result.json','pruning_plan.json','training_trace.jsonl','events.jsonl','run.log',
                                         'lora_recovery/recovery_result.json','lora_recovery/recovery_manifest.json',
                                         'lora_recovery/lora_layout.json','lora_recovery/finetuning_trace.jsonl',
                                         'lora_recovery/events.jsonl','lora_recovery/run.log'):
                            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
                            shutil.copy2(output/relative,target)


if __name__=='__main__':
    unittest.main()
