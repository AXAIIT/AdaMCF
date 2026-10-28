"""α 池持续取样；数据独立、随机序列独立，同一次差分评估固定批次。"""
from phi2_data import collate_examples, to_device


def collate_alpha_examples(rows, pad_token_id):
    return collate_examples(rows, pad_token_id), [row['row_id'] for row in rows]


class AlphaBatchStream:
    """持续遍历 shuffle=True 的 loader，遍历完后开始下一次随机排列。

    只把本次差分要用的批次搬到设备，避免把整个 α 池长期缓存到显存。
    一次 update 的正/负扰动必须复用同一列表，避免把数据波动当成 α 梯度。
    """

    def __init__(self, loader, device):
        if len(loader) == 0:
            raise ValueError('α loader 不能为空。')
        self.loader, self.device = loader, device
        self.iterator = iter(loader)
        self.used_row_ids = set()

    def draw(self, optimizer):
        cfg = optimizer.alpha_scheduler.cfg
        context = optimizer.alpha_update_context()
        # 与 CRIC 调度器的渐增微批规则一致；只改变来源，不改更新时间。
        count = min(len(self.loader), cfg.eval_batch_limit,
                    1 + max(0, context['round'] - cfg.start_after_period)
                    // max(1, cfg.eval_batch_grow_every))
        batches, row_ids = [], []
        for _ in range(count):
            try:
                item = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                item = next(self.iterator)
            # 原有测试/调用方的 dict loader 继续可用；正式 loader 另带原始行号。
            batch, ids = item if isinstance(item, tuple) else (item, [])
            batches.append((to_device(batch, self.device), None))
            row_ids.extend(ids)
        return batches, row_ids

    def commit(self, row_ids):
        # 调度器可能因 gap/stride 跳过更新，只有实际使用的数据才计入覆盖率。
        self.used_row_ids.update(row_ids)
        return len(self.used_row_ids)
