"""CRIC 的评分、索引、α 回滚和恢复训练回归；不下载模型或数据。"""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from only_train_once import OTO
from only_train_once.optimizer import HESSO, HESSOCRIC
from only_train_once.transform import TensorTransform as T, make_index_groups_transform

DEVICE = torch.device(os.environ.get('CRIC_TEST_DEVICE', 'cpu'))
torch.set_num_threads(2)
KEYS = ['l1_avg_magnitude', 'l2_avg_magnitude', 'cosine_similarity', 'fisher', 'grad_weight_product']


def param_group(p, transform=T.BASIC, count=4, name='group', prunable=True, auxiliary=False):
    return dict(id=name, params=[p], p_names=[name+'.weight'], p_transform=[transform],
                num_groups=count, is_prunable=prunable, is_auxiliary=auxiliary,
                auxiliary_ngs=[])


def optimizer(groups, **kwargs):
    settings=dict(lr=.01, start_cric_step=1, sampling_steps=2,
                  hybrid_training_steps=4, max_cycle_period=2,
                  alpha_scheduler_cfg={'warmup_updates':0, 'min_gap_std':1e9,
                                       'eval_batch_limit':1})
    settings.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return HESSOCRIC(groups, **settings)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(4,6)
        self.fc2=nn.Linear(6,2)

    def forward(self,x):
        return self.fc2(torch.relu(self.fc1(x)))


def network(variant='sgd', **kwargs):
    torch.manual_seed(17)
    model=Net().to(DEVICE)
    groups=[dict(id='hidden',params=[model.fc1.weight,model.fc1.bias,model.fc2.weight],
                 p_names=['fc1.weight','fc1.bias','fc2.weight'],
                 p_transform=[T.BASIC,T.BASIC,T.TRANSPOSE],num_groups=6,
                 is_prunable=True,is_auxiliary=False,auxiliary_ngs=[]),
            param_group(model.fc2.bias,count=2,name='output_bias',prunable=False)]
    return model,optimizer(groups,variant=variant,**kwargs)


def batch():
    g=torch.Generator().manual_seed(23)
    return torch.randn(8,4,generator=g).to(DEVICE),torch.arange(8,device=DEVICE)%2


def advance(model,opt,steps):
    x,y=batch();criterion=nn.CrossEntropyLoss()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            opt.zero_grad()
            loss=criterion(model(x),y);loss.backward()
            if opt.will_update_alpha_on_next_step():
                opt.alpha_scheduler.update(model,criterion,[(x,y)])
            opt.step(loss=loss.detach())
    return loss


class CRICAdaptationTests(unittest.TestCase):
    def assert_same(self,a,b):
        self.assertTrue(torch.equal(a,b),f'max diff: {(a-b).abs().max().item()}')

    def test_public_factory_and_default_five_scores(self):
        model,opt=network()
        oto=OTO();oto._graph=SimpleNamespace(get_param_groups=lambda:opt.param_groups)
        with contextlib.redirect_stdout(io.StringIO()):
            created=oto.hessocric(sampling_steps=2,hybrid_training_steps=4,
                                  alpha_scheduler_cfg={'warmup_updates':0})
        self.assertIsInstance(created,HESSOCRIC)
        self.assertEqual(created._alpha_keys,KEYS)
        self.assertEqual(created.importance_score_criteria['loss'],1.)
        x,y=batch();nn.CrossEntropyLoss()(model(x),y).backward()
        created.preview_grad_variant();created.compute_importance_scores()
        self.assertEqual(set(created.param_groups[0]['_raw_scores']),set(KEYS))
        self.assertTrue(torch.isfinite(torch.cat(created.global_scores)).all())

    def test_legacy_score_aliases_are_explicit(self):
        p=nn.Parameter(torch.ones(4,2,device=DEVICE))
        opt=optimizer([param_group(p)],importance_score_criteria={'magnitude':.4,'avg_magnitude':.6,'loss':2})
        self.assertEqual(opt._alpha_keys,['l2_magnitude','l2_avg_magnitude'])
        self.assertEqual(opt.importance_score_criteria['loss'],2)

    def test_invalid_config_fails_early(self):
        for kwargs in [dict(sampling_steps=0),dict(hybrid_training_steps=None),
                       dict(target_group_sparsity=1),dict(importance_score_criteria={'unknown':1}),
                       dict(alpha_scheduler_cfg={'lr':float('nan')})]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):network(**kwargs)

    def test_scaling_all_transform_contracts_and_exact_restore(self):
        cases=[(T.BASIC,(4,3),4,[1],0,[1]),
               (T.TRANSPOSE,(3,4),4,[1],1,[1]),
               (T.MULTIHEAD_HEADDIM,(8,3),4,[1],0,[1,5]),
               (T.MULTIHEAD_NUMHEAD,(8,3),4,[1],0,[2,3]),
               (make_index_groups_transform([[0,3],[1,4],[2,5]],axis=0),(6,2),3,[1],0,[1,4]),
               (make_index_groups_transform([[0,3],[1,4],[2,5]],axis=1),(2,6),3,[1],1,[1,4])]
        for transform,shape,count,logical,axis,physical in cases:
            with self.subTest(transform=str(transform)):
                p=nn.Parameter(torch.arange(1,1+int(torch.tensor(shape).prod()),device=DEVICE,dtype=torch.float32).reshape(shape))
                group=param_group(p,transform,count)
                if transform in (T.MULTIHEAD_HEADDIM,T.MULTIHEAD_NUMHEAD):
                    group.update(num_heads=2 if transform==T.MULTIHEAD_HEADDIM else 4,head_dim=4 if transform==T.MULTIHEAD_HEADDIM else 2)
                opt=optimizer([group]);before=p.detach().clone()
                saved=opt._apply_ratio_and_backup(group,logical,0)
                expected=before.clone();expected.index_fill_(axis,torch.tensor(physical,device=DEVICE),0)
                self.assert_same(p,expected)
                opt._restore_from_backups(saved);self.assert_same(p,before)
                opt._scale_group(group,logical,.5)
                expected=before.clone();idx=torch.tensor(physical,device=DEVICE)
                expected.index_copy_(axis,idx,before.index_select(axis,idx)*.5)
                self.assert_same(p,expected)

    def test_auxiliary_offset_and_no_prune_are_preserved(self):
        p=nn.Parameter(torch.ones(4,2,device=DEVICE));frozen=nn.Parameter(torch.ones(4,2,device=DEVICE))
        aux=nn.Parameter(torch.ones(8,device=DEVICE),requires_grad=False)
        group=param_group(p);group['params'].append(frozen);group['p_names'].append('frozen')
        group['p_transform'].append(T.NO_PRUNE);group['auxiliary_ngs']=[('aux',2)]
        opt=optimizer([group,param_group(aux,count=8,name='aux',prunable=False,auxiliary=True)])
        opt._scale_group(group,[1],0)
        self.assertEqual(aux[3].item(),0)
        self.assertEqual(aux.sum().item(),7)
        self.assertTrue(torch.equal(frozen,torch.ones_like(frozen)))
        group['pruned_idxes']=[2]
        opt.fix_pruned_groups_as_zeros(group)
        self.assertEqual(p[2].sum().item(),0)
        self.assertEqual(aux[4].item(),0)
        self.assertTrue(torch.equal(frozen,torch.ones_like(frozen)))

    def test_boundaries_and_no_interior_alpha_updates(self):
        model,opt=network(start_cric_step=3)
        self.assertIsNone(opt.alpha_update_context())
        opt.num_steps=2;self.assertEqual(opt.alpha_update_context()['round'],0)
        opt.num_steps=3;self.assertIsNone(opt.alpha_update_context())
        opt.num_steps=opt.start_global_sampling_step-1
        self.assertEqual(opt.alpha_update_context()['round'],1)
        opt.num_steps+=1;before=opt.alpha_scheduler.alpha.clone()
        self.assertFalse(opt.alpha_scheduler.update(model,None,None))
        self.assert_same(opt.alpha_scheduler.alpha,before)

    def test_final_alpha_refuses_to_mix_old_fused_scores(self):
        _,opt=network();group=opt.param_groups[0]
        group['raw_score_sums']={0:{k:torch.full((6,),float(i+1)) for i,k in enumerate(KEYS)},
                                 1:{k:torch.full((6,),float(2*(i+1))) for i,k in enumerate(KEYS)}}
        group['raw_score_counts']={0:1,1:1}
        final=[.4,.3,.1,.1,.1]
        for k,w in zip(KEYS,final):opt.importance_score_criteria[k]=w
        group['importance_score_collection'][0]=[{'overall':torch.full((6,),999.)}]
        group['active_violating_idxes_collection'][0]=[1]
        group['loss_collection'][0]=[2.]
        opt.compute_accumulate_saliency_score()
        expected=torch.full((6,),sum(1.5*(i+1)*w for i,w in enumerate(final)),device=DEVICE)
        expected[1]+=2.
        self.assertTrue(torch.allclose(group['accumulated_importance_score'],expected))

    def test_alpha_failure_restores_weights_modes_and_momentum(self):
        model,opt=network(variant='adamw');model.fc1.eval()
        x,y=batch();nn.CrossEntropyLoss()(model(x),y).backward()
        before=copy.deepcopy(model.state_dict());modes=[m.training for m in model.modules()]
        def broken(*_):raise TypeError('intentional loss adapter failure')
        with self.assertRaisesRegex(TypeError,'intentional'):
            opt.alpha_scheduler.update(model,broken,[(x,y)])
        for k,v in model.state_dict().items():self.assert_same(v,before[k])
        self.assertEqual([m.training for m in model.modules()],modes)
        self.assertFalse(opt.first_moment_grads);self.assertFalse(opt.second_moment_grads)
        self.assertEqual(opt.num_steps,0)

    def test_preview_does_not_advance_adam_and_matches_real_step(self):
        for variant in ['sgd','adam','adamw']:
            with self.subTest(variant=variant):
                model,opt=network(variant=variant,adaptive_alpha=False,start_cric_step=3,weight_decay=.013)
                advance(model,opt,1)
                x,y=batch();opt.zero_grad();loss=nn.CrossEntropyLoss()(model(x),y);loss.backward()
                before=copy.deepcopy(opt.first_moment_grads)
                opt.preview_grad_variant();preview=[{k:v.clone() for k,v in g['grad_variant'].items()} for g in opt.param_groups]
                for k,v in opt.first_moment_grads.items():self.assert_same(v,before[k])
                opt.step(loss.detach())
                for group,saved in zip(opt.param_groups,preview):
                    for k,v in group['grad_variant'].items():self.assert_same(v,saved[k])

    def test_full_cric_freezes_zero_groups_and_learns_alpha(self):
        model,opt=network();loss=advance(model,opt,20)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(opt.is_cric_terminated)
        self.assertGreaterEqual(opt.alpha_scheduler.update_count,2)
        self.assertTrue((opt.alpha_scheduler.alpha>0).all())
        self.assertAlmostEqual(opt.alpha_scheduler.alpha.sum().item(),1.,places=6)
        indexes=opt.param_groups[0]['pruned_idxes'];self.assertEqual(len(indexes),3)
        self.assertEqual(torch.count_nonzero(model.fc1.weight[indexes]).item(),0)
        self.assertEqual(torch.count_nonzero(model.fc2.weight[:,indexes]).item(),0)
        self.assertEqual(opt.importance_score_criteria['loss'],1.)
        opt.compute_metrics()

    def test_checkpoint_resume_matches_continuous_adam_training(self):
        for split in [2,8,10,17]:  # 投影、全局采样、混合稀疏化、剪后训练。
            with self.subTest(checkpoint_step=split):
                model,opt=network(variant='adamw');advance(model,opt,split)
                with tempfile.TemporaryDirectory() as tmp:
                    file=Path(tmp)/'own_test_checkpoint.pt'
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},file)
                    saved=torch.load(file,map_location=DEVICE,weights_only=False)
                advance(model,opt,20-split)
                other,resumed=network(variant='adamw');other.load_state_dict(saved['model']);resumed.load_state_dict(saved['optimizer'])
                for group in resumed.param_groups:
                    for scores in group.get('raw_score_sums',{}).values():
                        self.assertTrue(all(value.device.type=='cpu' for value in scores.values()))
                advance(other,resumed,20-split)
                for key,value in model.state_dict().items():self.assert_same(value,other.state_dict()[key])
                self.assert_same(opt.alpha_scheduler.alpha,resumed.alpha_scheduler.alpha)
                self.assertEqual(opt.num_steps,resumed.num_steps)
                self.assertEqual(opt.param_groups[0]['pruned_idxes'],resumed.param_groups[0]['pruned_idxes'])

    def test_partial_mask_application_failure_rolls_back(self):
        model,opt=network();before=model.fc1.weight.detach().clone()
        class BrokenParameter:
            @property
            def data(self):raise RuntimeError('intentional partial-mask failure')
        # 故意模拟掩码处理中途出错，验证“本组还未返回备份”也不会遗留修改。
        opt._pruning_slices=lambda *_:[
            (model.fc1.weight,0,torch.tensor([0],device=DEVICE)),
            (BrokenParameter(),0,torch.tensor([0],device=DEVICE))]
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            opt._apply_ratio_and_backup(opt.param_groups[0],[0],0)
        self.assert_same(model.fc1.weight,before)

    def test_mapping_inputs_with_model_loss(self):
        class MappingNet(Net):
            def forward(self,input_ids,labels):
                return {'loss':nn.functional.cross_entropy(super().forward(input_ids),labels)}
        model,opt=network();model.__class__=MappingNet
        x,y=batch();out=model(x,y);out['loss'].backward()
        self.assertTrue(opt.alpha_scheduler.update(model,None,[({'input_ids':x,'labels':y},None)]))

    def test_alpha_responds_to_loss_and_only_once_per_boundary(self):
        # 构造一个明确的排序边界：两个准则分别偏好剪第 0/1 行，
        # 而第 1 行对损失的贡献更大，剪掉它的评估损失更小。
        model=nn.Linear(1,4,bias=False).to(DEVICE)
        with torch.no_grad():model.weight.copy_(torch.tensor([[1.],[2.],[1.],[1.]],device=DEVICE))
        opt=optimizer([param_group(model.weight)],
                      importance_score_criteria={KEYS[0]:.5,KEYS[1]:.5})
        group=opt.param_groups[0];group['global_start_idx']=0
        def controlled_scores():
            group['_raw_scores']={KEYS[0]:torch.tensor([.49,.51,.1,.9],device=DEVICE),
                                  KEYS[1]:torch.tensor([.51,.49,.1,.9],device=DEVICE)}
            opt._reaggregate_overall_from_alpha(opt.alpha_scheduler.alpha)
        opt.compute_importance_scores=controlled_scores
        inputs=torch.ones(1,1,device=DEVICE)
        criterion=lambda output,_:output.square().mean()
        criterion(model(inputs),None).backward()
        before=model.weight.detach().clone()
        self.assertTrue(opt.alpha_scheduler.update(model,criterion,[(inputs,None)]))
        self.assertLess(opt.alpha_scheduler.alpha[0].item(),.5)
        self.assertGreater(opt.alpha_scheduler.alpha[1].item(),.5)
        self.assert_same(model.weight,before)
        self.assertFalse(opt.alpha_scheduler.update(model,criterion,[(inputs,None)]))
        self.assertEqual(opt.alpha_scheduler.update_count,1)

    def test_missing_loss_fails_without_mutation(self):
        model,opt=network();before=copy.deepcopy(model.state_dict())
        with self.assertRaises(ValueError):opt.step()
        self.assertEqual(opt.num_steps,0)
        for k,v in model.state_dict().items():self.assert_same(v,before[k])

    def test_training_fallback_restores_batchnorm_buffers(self):
        class BNNet(Net):
            def forward(self,x):return self.fc2(self.dropout(torch.relu(self.bn(self.fc1(x)))))
        model,opt=network();model.__class__=BNNet
        model.bn=nn.BatchNorm1d(6).to(DEVICE)
        model.dropout=nn.Dropout(.4)
        x,y=batch();nn.CrossEntropyLoss()(model(x),y).backward()
        before=copy.deepcopy(model.state_dict())
        rng=torch.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state(DEVICE).clone() if DEVICE.type=='cuda' else None
        def training_loss(output,target):
            if not model.training:raise ValueError('criterion requires train output')
            return nn.functional.cross_entropy(output,target)
        self.assertTrue(opt.alpha_scheduler.update(model,training_loss,[(x,y)]))
        for k,v in model.state_dict().items():self.assert_same(v,before[k])
        self.assert_same(torch.get_rng_state(),rng)
        if cuda_rng is not None:self.assert_same(torch.cuda.get_rng_state(DEVICE),cuda_rng)

    def test_random_small_phi_causal_loss_and_cric(self):
        from transformers import PhiConfig,PhiForCausalLM
        torch.manual_seed(31)
        config=PhiConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
                         num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
                         max_position_embeddings=32,partial_rotary_factor=.5)
        config._attn_implementation='eager';config.use_cache=False
        model=PhiForCausalLM(config).to(DEVICE)
        names={id(p):n for n,p in model.named_parameters()};groups=[];used=set()
        def add(identifier,params,transforms,count):
            groups.append(dict(id=identifier,params=params,p_names=[names[id(p)] for p in params],
                               p_transform=transforms,num_groups=count,is_prunable=True,
                               is_auxiliary=False,auxiliary_ngs=[]))
            used.update(id(p) for p in params)
        for index,layer in enumerate(model.model.layers):
            mlp=layer.mlp
            add(f'ffn{index}',[mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight],
                [T.BASIC,T.BASIC,T.TRANSPOSE],64)
            attn=layer.self_attn;blocks=[list(range(h*8,(h+1)*8)) for h in range(4)]
            params=[];transforms=[]
            for proj in [attn.q_proj,attn.k_proj,attn.v_proj]:
                params.extend([proj.weight,proj.bias]);transforms.extend([make_index_groups_transform(blocks)]*2)
            params.append(attn.dense.weight);transforms.append(make_index_groups_transform(blocks,axis=1))
            add(f'attention{index}',params,transforms,4)
        rest=[p for p in model.parameters() if id(p) not in used]
        groups.append(dict(id='remaining',params=rest,p_names=[names[id(p)] for p in rest],
                           p_transform=[T.NO_PRUNE]*len(rest),num_groups=1,is_prunable=False,
                           is_auxiliary=False,auxiliary_ngs=[]))
        opt=optimizer(groups,target_group_sparsity=.25,variant='adamw')
        tokens=torch.arange(8,device=DEVICE).reshape(1,-1)
        inputs={'input_ids':tokens,'labels':tokens.clone(),'use_cache':False}
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(36):
                opt.zero_grad();loss=model(**inputs).loss;loss.backward()
                if opt.will_update_alpha_on_next_step():opt.alpha_scheduler.update(model,None,[(inputs,None)])
                opt.step(loss.detach())
        self.assertTrue(torch.isfinite(loss));self.assertTrue(opt.is_cric_terminated)
        self.assertGreaterEqual(opt.alpha_scheduler.update_count,2)
        self.assertEqual(sum(len(g['pruned_idxes']) for g in opt.param_groups),34)
        self.assertTrue(torch.isfinite(model(**inputs).logits).all())

    def test_no_pruning_baseline(self):
        model,opt=network(target_group_sparsity=0,variant='adamw')
        self.assertTrue(torch.isfinite(advance(model,opt,3)))
        self.assertEqual(opt.num_steps,3);self.assertEqual(opt.alpha_scheduler.update_count,0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CRICAdaptationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print('CRIC_TEST_SUMMARY',json.dumps({'device':str(DEVICE),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
