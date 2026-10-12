"""The decision model: a causal LM backbone (optionally with LoRA) read out at answer slots in one forward pass."""
import json
import os

import torch
import torch.nn as nn

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class JevModel(nn.Module):
    """Choice questions use the LM head restricted to the option-letter tokens; pointer questions use a
    small bilinear head between the answer slot and each block marker, plus a learned "none" key."""

    def __init__(self, lm, ptr_dim=256):
        super().__init__()
        self.lm = lm
        hidden = self.causal_lm().config.get_text_config().hidden_size
        # LM hidden states have a few huge dimensions; normalize them so pointer logits start near zero
        self.ptr_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ptr_q = nn.Linear(hidden, ptr_dim, bias=False)
        self.ptr_k = nn.Linear(hidden, ptr_dim, bias=False)
        self.ptr_none = nn.Parameter(torch.zeros(ptr_dim))
        nn.init.zeros_(self.ptr_q.weight)  # the untrained head is exactly uniform over blocks and "none"
        nn.init.normal_(self.ptr_k.weight, std=hidden ** -0.5)
        self.ptr_scale = ptr_dim ** -0.5

    def causal_lm(self):
        return self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm

    def heads(self):
        return [self.ptr_q.weight, self.ptr_k.weight, self.ptr_none]

    def forward(self, input_ids, attention_mask, questions):
        """questions: one list per sequence of {"type", "pos", "option_ids" | "marker_pos"}.
        Returns one list of float32 logit vectors per sequence."""
        clm = self.causal_lm()
        hidden = clm.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        head = clm.get_output_embeddings().weight
        out = []
        for b, qs in enumerate(questions):
            row = []
            for q in qs:
                h = hidden[b, q["pos"]].float()
                if q["type"] == "choice":
                    ids = torch.tensor(q["option_ids"], device=hidden.device)
                    row.append(head[ids].float() @ h)
                else:
                    pos = torch.tensor(q["marker_pos"], device=hidden.device)
                    query = self.ptr_q(self.ptr_norm(h))
                    keys = self.ptr_k(self.ptr_norm(hidden[b, pos].float()))
                    row.append(torch.cat([keys @ query, (self.ptr_none @ query).unsqueeze(0)]) * self.ptr_scale)
            out.append(row)
        return out


def soft_cross_entropy(logits, target):
    """Cross-entropy against a target distribution; minimized by the target itself (a proper scoring rule)."""
    return -(target * torch.log_softmax(logits, dim=-1)).sum()


def load_lm(model_name, device, attn="sdpa"):
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attn)
    lm.config.use_cache = False
    return lm


def add_lora(lm, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                     task_type="CAUSAL_LM")
    return get_peft_model(lm, cfg)


def save_checkpoint(model, out_dir, run_config):
    """LoRA adapter (peft format) + pointer head + the settings needed to rebuild inputs."""
    os.makedirs(out_dir, exist_ok=True)
    model.lm.save_pretrained(os.path.join(out_dir, "adapter"))
    torch.save({"ptr_q": model.ptr_q.state_dict(), "ptr_k": model.ptr_k.state_dict(),
                "ptr_none": model.ptr_none.detach().cpu()}, os.path.join(out_dir, "heads.pt"))
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)


def load_for_inference(model_name, checkpoint, device, attn="sdpa"):
    """Base model plus a checkpoint from save_checkpoint, or the bare base model (zero-shot) if checkpoint is None.
    The untrained pointer head predicts uniformly, so zero-shot pointer metrics are a chance baseline."""
    lm = load_lm(model_name, device, attn)
    if checkpoint:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, os.path.join(checkpoint, "adapter"))
    model = JevModel(lm)
    if checkpoint:
        heads = torch.load(os.path.join(checkpoint, "heads.pt"), map_location="cpu")
        model.ptr_q.load_state_dict(heads["ptr_q"])
        model.ptr_k.load_state_dict(heads["ptr_k"])
        model.ptr_none.data.copy_(heads["ptr_none"])
    return model.to(device).eval()
