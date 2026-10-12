"""Rendered examples as a torch dataset."""
import json

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


def collate(batch, pad_id):
    """Right-pad to the longest sequence; readout positions are unaffected by right padding."""
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
        mask[i, :len(x["input_ids"])] = 1
    return {"input_ids": ids, "attention_mask": mask, "items": batch}
