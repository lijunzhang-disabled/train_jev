"""Rendered examples as a torch dataset."""
import collections
import json
import math
import random

import torch

from jev.format import Renderer


class JevDataset(torch.utils.data.Dataset):
    def __init__(self, path, renderer: Renderer, limit=None):
        self.items, self.dropped = [], 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and len(self.items) >= limit:
                    break
                item = renderer.render(json.loads(line))
                if item is None:
                    self.dropped += 1
                else:
                    self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def tokens(self):
        return sum(len(x["input_ids"]) for x in self.items)


class GroupSampler(torch.utils.data.Sampler):
    """Shuffles groups instead of examples, keeping each group's examples next to each other. A bug's buggy,
    fixed and elsewhere versions are near-identical files with opposite labels; trained in different steps they
    push the model back and forth on the shared text, while in one step only the lines that differ get the
    update. Rank r takes every world-th example of the shared order, so a group lands in the same micro-batch
    across processes (or the next one), within one optimizer step unless it straddles a step boundary."""

    def __init__(self, dataset, rank=0, world=1, seed=0):
        groups = collections.defaultdict(list)
        for i, item in enumerate(dataset.items):
            groups[item.get("group", item["id"])].append(i)
        self.groups = [groups[k] for k in sorted(groups)]
        self.n = len(dataset.items)
        self.rank, self.world, self.seed, self.epoch = rank, world, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        groups = list(self.groups)
        random.Random(self.seed * 100003 + self.epoch).shuffle(groups)
        order = [i for g in groups for i in g]
        order += order[:len(self) * self.world - len(order)]  # pad like DistributedSampler: every rank gets len(self)
        return iter(order[self.rank::self.world])

    def __len__(self):
        return math.ceil(self.n / self.world)


def collate(batch, pad_id):
    """Right-pad to the longest sequence; readout positions are unaffected by right padding."""
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
        mask[i, :len(x["input_ids"])] = 1
    return {"input_ids": ids, "attention_mask": mask, "items": batch}
