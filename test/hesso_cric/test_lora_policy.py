"""LoRA 在 CRIC 中统一排除；验证真实评分、梯度和辅助传播路径。"""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from only_train_once.optimizer import HESSO
from only_train_once.transform import TensorTransform as T
from test_adamcf_cric import optimizer, param_group, DEVICE


def mixed_group():
    base = nn.Parameter(torch.arange(1, 13, device=DEVICE, dtype=torch.float32).reshape(4, 3))
    # A 的 rank 与基座行数不同，能捕获错误地让 A 参与评分/逐行裁剪的情况。
    a = nn.Parameter(torch.ones(2, 3, device=DEVICE))
    b = nn.Parameter(torch.ones(4, 2, device=DEVICE))
    group = param_group(base)
    group.update(params=[base, a, b],
                 p_names=['layer.weight', 'layer.lora_A.default.weight', 'layer.lora_B.default.weight'],
                 p_transform=[T.BASIC, T.BASIC, T.BASIC])
    return group, (base, a, b)


class LoRAPolicyTests(unittest.TestCase):
    def test_mixed_group_excludes_scores_but_retains_normal_gradients(self):
        source, (base, a, b) = mixed_group()
        opt = optimizer([source], start_cric_step=100)
        reference = optimizer([param_group(nn.Parameter(base.detach().clone()))], start_cric_step=100)
        # 算子原始分组没有被改写；优化器内部保护 A/B。
        self.assertEqual(source['p_transform'], [T.BASIC] * 3)
        self.assertEqual(opt.param_groups[0]['p_transform'], [T.BASIC, T.NO_PRUNE, T.NO_PRUNE])
        for candidate in (opt, reference):
            for group in candidate.param_groups:
                for p in group['params']:
                    p.grad = torch.ones_like(p)
            candidate.preview_grad_variant()
            candidate.compute_importance_scores()
            candidate.compute_metrics()
        for key in opt._alpha_keys:
            torch.testing.assert_close(opt.param_groups[0]['_raw_scores'][key],
                                       reference.param_groups[0]['_raw_scores'][key], rtol=0, atol=0)
        before = [p.detach().clone() for p in (a, b)]
        opt.step(loss=torch.tensor(1., device=DEVICE))
        for p, previous in zip((a, b), before):
            torch.testing.assert_close(p, previous - .01)
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(b.shape, (4, 2))

    def test_trial_final_zeroing_and_auxiliary_paths_preserve_lora(self):
        source, (base, a, b) = mixed_group()
        aux = nn.Parameter(torch.ones(6, 2, device=DEVICE))
        aux_lora = nn.Parameter(torch.ones(6, 2, device=DEVICE))
        aux_group = param_group(aux, count=6, name='aux', prunable=False, auxiliary=True)
        aux_group.update(params=[aux, aux_lora], p_names=['aux.weight', 'aux.lora_embedding_A'],
                         p_transform=[T.BASIC, T.BASIC])
        source['auxiliary_ngs'] = [('aux', 1)]
        opt = optimizer([source, aux_group])
        group = opt.param_groups[0]
        protected_before = [p.detach().clone() for p in (a, b, aux_lora)]
        backup = opt._apply_ratio_and_backup(group, [1], 0)
        self.assertEqual(base[1].count_nonzero(), 0)
        self.assertEqual(aux[2].count_nonzero(), 0)
        opt._restore_from_backups(backup)
        group['pruned_idxes'] = [1]
        opt.fix_pruned_groups_as_zeros(group)
        opt._scale_group(group, [2], .25)
        for p, previous in zip((a, b, aux_lora), protected_before):
            torch.testing.assert_close(p, previous, rtol=0, atol=0)

    def test_lora_only_groups_never_consume_budget_and_preserve_no_update(self):
        base = nn.Parameter(torch.ones(4, 2, device=DEVICE))
        a = nn.Parameter(torch.ones(4, 2, device=DEVICE))
        b = nn.Parameter(torch.ones(4, 2, device=DEVICE))
        ga = param_group(a, name='layer.lora_A')
        gb = param_group(b, transform=T.NO_UPDATE, name='layer.LoRA_magnitude_vector')
        opt = optimizer([param_group(base), ga, gb])
        self.assertEqual(opt.total_num_groups, 4)
        self.assertEqual(opt.target_num_redundant_groups, 2)
        self.assertFalse(opt.param_groups[1]['is_prunable'])
        self.assertFalse(opt.param_groups[2]['is_prunable'])
        self.assertEqual(opt.param_groups[2]['p_transform'], [T.NO_UPDATE])

    def test_statistics_restore_parameter_metadata_after_failure(self):
        source, _ = mixed_group()
        opt = optimizer([source])
        original = opt.param_groups[0]['params']
        with patch.object(HESSO, 'compute_importance_scores', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                opt.compute_importance_scores()
        self.assertIs(opt.param_groups[0]['params'], original)
        self.assertEqual(len(original), 3)

    def test_checkpoint_keeps_protection_and_rejects_unsafe_legacy_policy(self):
        source, _ = mixed_group()
        opt = optimizer([source])
        state = copy.deepcopy(opt.state_dict())
        fresh, _ = mixed_group()
        restored = optimizer([fresh])
        restored.load_state_dict(state)
        self.assertEqual(restored.param_groups[0]['p_transform'][1:], [T.NO_PRUNE] * 2)
        state['param_groups'][0]['p_transform'][1] = T.BASIC
        with self.assertRaisesRegex(ValueError, '未保护 LoRA'):
            restored.load_state_dict(state)
        self.assertEqual(restored.param_groups[0]['p_transform'][1:], [T.NO_PRUNE] * 2)
        only = param_group(nn.Parameter(torch.ones(4, 2, device=DEVICE)), name='layer.lora_A')
        only_opt = optimizer([only])
        unsafe = copy.deepcopy(only_opt.state_dict())
        unsafe['param_groups'][0]['is_prunable'] = True
        with self.assertRaisesRegex(ValueError, '预算不兼容'):
            only_opt.load_state_dict(unsafe)


if __name__ == '__main__':
    unittest.main()
