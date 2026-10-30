"""公共 α 策略回归，不依赖 Phi 或任何模型专属脚本。"""
import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
from only_train_once.optimizer import (AlphaBatch, AlphaEvaluationPool, CRICAlphaPool,
    CRICAlphaScheduleConfig, prepare_cric_alpha_data)
from only_train_once.optimizer.alpha_data import select_alpha_batches, commit_alpha_selection
from only_train_once.optimizer.alpha_scheduler import AlphaScheduleConfig, LossGuidedAlphaScheduler
from only_train_once.optimizer.cric_alpha import CRICAlphaScheduler
from only_train_once.optimizer.cric_alpha import make_cric_alpha_config


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([.2,-.4,.7,1.1]))
        self.seen=[]

    def forward(self,x):
        self.seen.append(x.detach().clone())
        return (x*self.weight).sum(-1)


class Backend:
    """两个优化器的同一评估场景：预算/轮次相等，屏蔽不同算法的时间轴。"""
    def __init__(self,model):
        self.model=model;self.num_steps=2;self.start_pruning_step=1;self.curr_pruning_period=0
        self.active_num_redundant_groups=[1]*10;self.pruned_group_idxes=[];self.target_num_redundant_groups=1
        self.importance_score_criteria={'a':.5,'b':.5}
        self.param_groups=[{'params':[model.weight],'is_prunable':True,'is_auxiliary':False}]
        self.bank=torch.tensor([[.1,.9],[.9,.1],[.8,.8],[.6,.6]])
        self.compute_importance_scores()

    def alpha_update_context(self):
        return {'step':self.num_steps+1,'round':self.curr_pruning_period,'warmup':self.num_steps<self.start_pruning_step}

    def compute_grad_variant(self):pass
    def preview_grad_variant(self):pass
    def compute_importance_scores(self):
        self.global_scores=[self.bank@torch.tensor(list(self.importance_score_criteria.values()))]
    def _reaggregate_overall_from_alpha(self,alpha):
        self.importance_score_criteria=dict(zip(('a','b'),alpha.tolist()));self.global_scores=[self.bank@alpha]
    def _dry_pick_active_by_alpha(self,alpha):
        return {id(self.param_groups[0]):[int((self.bank@alpha).argmin())]}
    def _apply_ratio_and_backup(self,group,indexes,ratio):
        saved=self.model.weight[indexes].detach().clone()
        self.model.weight.data[indexes]*=ratio
        return [(indexes,saved)]
    def _restore_from_backups(self,backups):
        for indexes,saved in reversed(backups):self.model.weight.data[indexes]=saved


def loader(seed=81):
    g=torch.Generator().manual_seed(39)
    rows=[AlphaBatch(torch.randn(2,4,generator=g),torch.randn(2,generator=g),(i,),2) for i in range(6)]
    return DataLoader(rows,batch_size=None,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=lambda item:item)


class SharedAlphaTests(unittest.TestCase):
    def test_same_numerical_and_sampling_policy_for_hesso_and_cric(self):
        configurations=[{},dict(momentum=.8,g_ema_gamma=0,ema_gamma=0,max_step_l1=None),
                        dict(momentum=.7,lr=.8,max_step_l1=.01,ema_gamma=.3)]
        for settings in configurations:
            with self.subTest(settings=settings):
                config=AlphaScheduleConfig(min_gap_std=1e9,eval_batch_limit=3,**settings)
                models=[ToyModel(),ToyModel()];backends=[Backend(m) for m in models]
                schedulers=[LossGuidedAlphaScheduler(['a','b'],device='cpu',cfg=config),
                            CRICAlphaScheduler(['a','b'],device='cpu',cfg=config)]
                pools=[AlphaEvaluationPool(loader()),AlphaEvaluationPool(loader())]
                for scheduler,backend in zip(schedulers,backends):scheduler.attach_backend(backend)
                for period in range(5):
                    for scheduler,backend,model,pool in zip(schedulers,backends,models,pools):
                        backend.curr_pruning_period=period;backend.num_steps=period+2
                        scheduler.update(model,nn.MSELoss(),pool)
                    for key in ('alpha','m','g_ema'):
                        self.assertTrue(torch.equal(getattr(schedulers[0],key),getattr(schedulers[1],key)),key)
                    self.assertEqual(schedulers[0].last_eval_stats,schedulers[1].last_eval_stats)
                self.assertEqual(len(pools[0].used_sample_ids),6)
                for left,right in zip(models[0].seen,models[1].seen):self.assertTrue(torch.equal(left,right))
                self.assertTrue(torch.equal(models[0].weight,models[1].weight))

    def test_pool_random_order_covers_all_rows_and_preserves_global_rng(self):
        rng=torch.get_rng_state().clone();cfg=AlphaScheduleConfig(eval_batch_limit=1)
        pools=[AlphaEvaluationPool(loader()),AlphaEvaluationPool(loader())];orders=[]
        for pool in pools:
            order=[]
            for _ in range(6):
                selected=select_alpha_batches(pool,None,0,cfg,'cpu',0)
                order.extend(selected[0].sample_ids);commit_alpha_selection(pool,selected)
            orders.append(order)
        self.assertEqual(orders[0],orders[1]);self.assertEqual(set(orders[0]),set(range(6)))
        self.assertNotEqual(orders[0],list(range(6)))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_fixed_legacy_cache_keeps_cursor_and_growth(self):
        batches=[(torch.full((1,2),float(i)),torch.tensor([i])) for i in range(4)]
        cfg=AlphaScheduleConfig(eval_batch_limit=3,eval_batch_grow_every=2)
        selected=select_alpha_batches(batches,None,4,cfg,'cpu',3)
        self.assertEqual([int(b.targets[0]) for b in selected],[3,0,1])
        self.assertEqual(commit_alpha_selection(batches,selected)['microbatches'],3)

    def test_skipped_gap_does_not_consume_pool_or_count_coverage(self):
        for cls in (LossGuidedAlphaScheduler,CRICAlphaScheduler):
            with self.subTest(scheduler=cls.__name__):
                model=ToyModel();backend=Backend(model)
                backend.bank=torch.tensor([[0.,0.],[10.,10.],[20.,20.],[30.,30.]])
                cfg=AlphaScheduleConfig(min_gap_std=0.)
                scheduler=cls(['a','b'],device='cpu',cfg=cfg);scheduler.attach_backend(backend)
                pool=AlphaEvaluationPool([AlphaBatch(torch.ones(1,4),torch.zeros(1),(7,))])
                scheduler.update(model,nn.MSELoss(),pool)
                self.assertFalse(pool.used_sample_ids);self.assertIsNone(scheduler.last_eval_stats)
                self.assertEqual(pool.take(1)[0].sample_ids,(7,));self.assertFalse(model.seen)

    def test_all_finite_difference_probes_share_one_selected_cohort(self):
        model=ToyModel();backend=Backend(model)
        cfg=AlphaScheduleConfig(min_gap_std=1e9,eval_batch_limit=1)
        scheduler=CRICAlphaScheduler(['a','b'],device='cpu',cfg=cfg);scheduler.attach_backend(backend)
        pool=AlphaEvaluationPool(loader());scheduler.update(model,nn.MSELoss(),pool)
        self.assertEqual(len(model.seen),4)
        self.assertTrue(all(torch.equal(model.seen[0],item) for item in model.seen))
        self.assertEqual(len(pool.used_sample_ids),1)

    def test_failure_does_not_commit_coverage_and_cric_restores_parameters(self):
        model=ToyModel();backend=Backend(model)
        scheduler=CRICAlphaScheduler(['a','b'],device='cpu',cfg=AlphaScheduleConfig(min_gap_std=1e9))
        scheduler.attach_backend(backend);pool=AlphaEvaluationPool(loader())
        before=model.weight.detach().clone();alpha=scheduler.alpha.clone()
        def broken(*_):raise TypeError('broken criterion')
        with self.assertRaisesRegex(TypeError,'broken'):scheduler.update(model,broken,pool)
        self.assertTrue(torch.equal(model.weight,before));self.assertTrue(torch.equal(scheduler.alpha,alpha))
        self.assertFalse(pool.used_sample_ids)


class CRICRatioTests(unittest.TestCase):
    @staticmethod
    def dataset(size):
        return [(torch.full((4,),float(i)/100), torch.tensor(float(i)/100)) for i in range(size)]

    def test_single_task_fallback_defaults_and_disjoint_holdout(self):
        rows=self.dataset(1000)
        training,pool,info=prepare_cric_alpha_data(rows,batch_size=8)
        self.assertEqual((len(training),len(pool)),(872,128))
        self.assertFalse(set(training.indices)&set(pool.dataset.indices))
        self.assertEqual(info['source'],'training_holdout')
        selected=pool.take(CRICAlphaScheduleConfig())
        self.assertEqual(sum(len(b.sample_ids) for b in selected),64)
        self.assertEqual(len(selected),8)

    def test_explicit_multitask_pool_is_preserved_and_two_updates_cover_it(self):
        rows=self.dataset(448);labels=[f'task{i//64}' for i in range(448)]
        training,pool,info=prepare_cric_alpha_data(self.dataset(100),alpha_dataset=rows,
            alpha_groups=labels,batch_size=8)
        self.assertEqual((len(training),len(pool)),(100,448))
        self.assertIsNone(info['actual_pool_ratio'])
        cohorts=[]
        for _ in range(2):
            selected=pool.take(CRICAlphaScheduleConfig())
            ids=[i for b in selected for i in b.sample_ids];cohorts.append(set(ids))
            self.assertEqual((len(ids),len(set(ids)),len(selected)),(224,224,28))
            self.assertEqual(set(pool.last_selection['selected_by_group'].values()),{32})
            commit_alpha_selection(pool,selected)
        self.assertFalse(cohorts[0]&cohorts[1])
        self.assertEqual(len(pool.used_sample_ids),448)

    def test_training_fallback_supports_arbitrary_group_counts_and_shared_mode(self):
        rows=self.dataset(200);groups=['a']*100+['b']*100
        training,pool,info=prepare_cric_alpha_data(rows,train_groups=groups,holdout=False)
        self.assertIs(training,rows)
        self.assertEqual(info['source'],'training_shared')
        self.assertEqual(info['pool_by_group'],{'a':8,'b':8})
        self.assertEqual(len(pool),16)

    def test_unaligned_ratio_bounds_and_small_dataset(self):
        cfg=CRICAlphaScheduleConfig(align_power_of_two=False,pool_ratio=.1,
            update_ratio=.5,min_update_samples=3,max_update_samples=4)
        _,pool,_=prepare_cric_alpha_data(self.dataset(101),config=cfg)
        self.assertEqual(len(pool),11)
        self.assertEqual(sum(len(b.sample_ids) for b in pool.take(cfg)),4)
        _,tiny,_=prepare_cric_alpha_data(self.dataset(2))
        self.assertEqual(len(tiny),1)
        self.assertEqual(len(tiny.take(cfg)[0].sample_ids),1)
        with self.assertRaises(ValueError):prepare_cric_alpha_data(self.dataset(1))

    def test_no_duplicates_when_rounding_crosses_an_epoch(self):
        cfg=CRICAlphaScheduleConfig(align_power_of_two=False,update_ratio=.6)
        pool=CRICAlphaPool(self.dataset(7),batch_size=3)
        seen=set()
        for _ in range(6):
            ids=[i for b in pool.take(cfg) for i in b.sample_ids]
            self.assertEqual((len(ids),len(set(ids))),(5,5));seen.update(ids)
        self.assertEqual(len(seen),7)

    def test_local_rng_determinism_and_checkpoint_roundtrip(self):
        before=torch.get_rng_state().clone()
        pools=[CRICAlphaPool(self.dataset(13),batch_size=3,seed=51) for _ in range(2)]
        cfg=CRICAlphaScheduleConfig(align_power_of_two=False)
        first=pools[0].take(cfg);commit_alpha_selection(pools[0],first)
        pools[1].load_state_dict(pools[0].state_dict())
        self.assertEqual([b.sample_ids for b in pools[0].take(cfg)],
                         [b.sample_ids for b in pools[1].take(cfg)])
        self.assertEqual(pools[0].used_sample_ids,pools[1].used_sample_ids)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_ratio_scheduler_uses_same_samples_for_every_probe_and_restores_on_error(self):
        model=ToyModel();backend=Backend(model)
        cfg=CRICAlphaScheduleConfig(min_gap_std=1e9,align_power_of_two=False)
        scheduler=CRICAlphaScheduler(['a','b'],device='cpu',cfg=cfg);scheduler.attach_backend(backend)
        pool=CRICAlphaPool(self.dataset(12),batch_size=2)
        snapshot=pool.state_dict();weight=model.weight.detach().clone()
        def broken(*_):raise TypeError('ratio probe failed')
        with self.assertRaises(TypeError):scheduler.update(model,broken,pool)
        self.assertEqual(pool.cursors,snapshot['cursors'])
        self.assertEqual(pool.orders,snapshot['orders']);self.assertFalse(pool.used_sample_ids)
        self.assertTrue(torch.equal(weight,model.weight));model.seen=[]
        scheduler.update(model,nn.MSELoss(),pool)
        stats=scheduler.last_eval_stats
        self.assertEqual((stats['selected_samples'],stats['microbatches']),(6,3))
        self.assertEqual(len(model.seen),12)
        for index,item in enumerate(model.seen):self.assertTrue(torch.equal(item,model.seen[index%3]))

    def test_collate_failure_rolls_back_sampling_cursor(self):
        def broken(_):raise ValueError('collate failed')
        pool=CRICAlphaPool(self.dataset(8),collate_fn=broken)
        snapshot=pool.state_dict();model=ToyModel();backend=Backend(model)
        scheduler=CRICAlphaScheduler(['a','b'],device='cpu',cfg=CRICAlphaScheduleConfig(min_gap_std=1e9))
        scheduler.attach_backend(backend)
        with self.assertRaisesRegex(ValueError,'collate failed'):
            scheduler.update(model,nn.MSELoss(),pool)
        self.assertEqual(pool.cursors,snapshot['cursors'])

    def test_invalid_configuration_and_legacy_checkpoint_policy(self):
        for values in ({'pool_ratio':0},{'update_ratio':1.1},{'update_ratio':float('nan')},
                       {'min_update_samples':0},{'max_pool_samples':0},{'align_power_of_two':1}):
            with self.subTest(values=values),self.assertRaises(ValueError):CRICAlphaScheduleConfig(**values)
        self.assertEqual(make_cric_alpha_config().sampling_policy,'ratio')
        self.assertEqual(make_cric_alpha_config(AlphaScheduleConfig()).sampling_policy,'legacy')
        self.assertEqual(make_cric_alpha_config({'lr':.02},checkpoint=True).sampling_policy,'legacy')
        with self.assertRaises(ValueError):
            CRICAlphaPool(self.dataset(10),groups=['a'])

    def test_scheduler_checkpoint_restores_attached_pool_before_next_update(self):
        cfg=CRICAlphaScheduleConfig(min_gap_std=1e9,align_power_of_two=False)
        def setup():
            model=ToyModel();backend=Backend(model)
            scheduler=CRICAlphaScheduler(['a','b'],device='cpu',cfg=cfg);scheduler.attach_backend(backend)
            return model,backend,scheduler,CRICAlphaPool(self.dataset(12),batch_size=2)
        left,lb,ls,lp=setup();ls.update(left,nn.MSELoss(),lp);saved=ls.state_dict()
        right,rb,rs,rp=setup();rs.load_state_dict(saved)
        lb.num_steps=rb.num_steps=3
        ls.update(left,nn.MSELoss(),lp);rs.update(right,nn.MSELoss(),rp)
        self.assertEqual(ls.last_eval_stats,rs.last_eval_stats)
        self.assertTrue(torch.equal(ls.alpha,rs.alpha))

    def test_ratio_pool_gap_skip_does_not_consume_rows(self):
        model=ToyModel();backend=Backend(model)
        backend.bank=torch.tensor([[0.,0.],[10.,10.],[20.,20.],[30.,30.]])
        scheduler=CRICAlphaScheduler(['a','b'],device='cpu',cfg=CRICAlphaScheduleConfig(min_gap_std=0))
        scheduler.attach_backend(backend);pool=CRICAlphaPool(self.dataset(12))
        snapshot=pool.state_dict()
        self.assertFalse(scheduler.update(model,nn.MSELoss(),pool))
        self.assertEqual(pool.cursors,snapshot['cursors'])
        self.assertEqual(scheduler.last_skip_reason,'confident_boundary')

    def test_invalid_pool_restore_and_legacy_source_are_explicit(self):
        pool=CRICAlphaPool(self.dataset(12),batch_size=2)
        with self.assertRaises(ValueError):
            CRICAlphaPool(self.dataset(12),batch_size=3).load_state_dict(pool.state_dict())
        with self.assertRaises(ValueError):
            CRICAlphaPool(self.dataset(2),groups=[1,'1'])
        with self.assertRaises(ValueError):
            prepare_cric_alpha_data(self.dataset(12),config={'sampling_policy':'legacy'})

    def test_real_optimizer_checkpoint_keeps_ratio_config_and_pool_progress(self):
        from test_adamcf_cric import network, batch
        inputs,targets=batch();rows=list(zip(inputs,targets));criterion=nn.CrossEntropyLoss()
        def setup():
            model,opt=network(variant='adamw')
            _,pool,_=opt.prepare_alpha_data(rows,alpha_dataset=rows,batch_size=2,seed=51)
            return model,opt,pool
        def advance(model,opt,count):
            for _ in range(count):
                opt.zero_grad();loss=criterion(model(inputs),targets);loss.backward()
                if opt.will_update_alpha_on_next_step():opt.alpha_scheduler.update(model,criterion)
                opt.step(loss.detach())
        left,lo,lp=setup();advance(left,lo,2)
        model_state=copy.deepcopy(left.state_dict());state=copy.deepcopy(lo.state_dict())
        right,ro,rp=setup();right.load_state_dict(model_state);ro.load_state_dict(state)
        advance(left,lo,18);advance(right,ro,18)
        self.assertEqual(ro.alpha_scheduler.cfg.update_ratio,.5)
        self.assertEqual(ro.alpha_scheduler.cfg.sampling_policy,'ratio')
        for key,value in left.state_dict().items():self.assertTrue(torch.equal(value,right.state_dict()[key]))
        self.assertEqual(lp.used_sample_ids,rp.used_sample_ids)
        self.assertTrue(torch.equal(lo.alpha_scheduler.alpha,ro.alpha_scheduler.alpha))


if __name__=='__main__':unittest.main()
