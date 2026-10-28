"""公共 α 策略回归，不依赖 Phi 或任何模型专属脚本。"""
import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
from only_train_once.optimizer import AlphaBatch, AlphaEvaluationPool
from only_train_once.optimizer.alpha_data import select_alpha_batches, commit_alpha_selection
from only_train_once.optimizer.alpha_scheduler import AlphaScheduleConfig, LossGuidedAlphaScheduler
from only_train_once.optimizer.cric_alpha import CRICAlphaScheduler


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


if __name__=='__main__':unittest.main()
