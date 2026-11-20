#!/usr/bin/env python3
"""test_drafter_masking.py -- executed test of the drafter's attention masking in fastrl@bce3df7.

Loads fastrl's own drafter class (verl/workers/drafter/model/qwen2_eagle.py,
Qwen2ForCausalLMEagle) by file path, instantiates it with a tiny random
configuration on CPU, and feeds it inputs shaped exactly as the Spot Trainer
builds them (eagle_background_trainer.py:410-417, 504-509): concatenated
samples, an all-ones attention mask, and no position_ids.

Experiments (all deterministic, fp32, CPU):
  C0  control      : the same input twice gives identical outputs.
  C1  causality    : change the LAST token of a sample; do outputs at EARLIER
                     positions change? (They must not, in a causal model.)
  C2  cross-sample : output for sample B alone vs. B packed after sample A.
  C3  attribution  : C2 repeated with position_ids reset at the boundary, which
                     separates attention leakage from the rotary-position shift.

What this test does NOT establish: any effect on accept length or end-to-end
speed. Those require training runs on real data and GPUs.

Usage:
    python test_drafter_masking.py /path/to/fastrl      # fastrl checked out at bce3df7
Requirements: torch (CPU is fine) and transformers; fastrl's runtime pins
transformers==4.57.1 (third-party/sglang/python/pyproject.toml), so test with that.
"""
import importlib.util
import os
import platform
import sys

import torch
import transformers
from transformers import Qwen2Config

FASTRL = sys.argv[1] if len(sys.argv) > 1 else "fastrl"
MODEL_FILE = os.path.join(FASTRL, "verl", "workers", "drafter", "model", "qwen2_eagle.py")

spec = importlib.util.spec_from_file_location("fastrl_qwen2_eagle", MODEL_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LA, LB, TOL = 12, 10, 1e-5


def build(requested_impl):
    """Build the drafter as fsdp_workers.py:500-518 does: a copy of the actor config with one layer.
    The actor is loaded with attn_implementation="flash_attention_2" (fsdp_workers.py:275); the
    drafter class then swaps flash_attention_2 for eager (qwen2_eagle.py:193-197)."""
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      use_cache=False)
    cfg._attn_implementation = requested_impl
    m = mod.Qwen2ForCausalLMEagle(cfg).float().eval()
    torch.nn.init.normal_(m.model.fc.weight, std=0.05)  # fc is created without post_init
    return m


@torch.no_grad()
def run(model, ids, hs, position_ids=None):
    """Call the drafter exactly as the trainer does: all-ones mask, no position_ids."""
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), base_model_hidden_states=hs,
                position_ids=position_ids, output_hidden_states=True)
    h = out.hidden_states[-1]
    return h[0] if h.dim() == 3 else h  # [seq, hidden]


def diff(x, y):
    d = (x - y).abs()
    return float("nan") if torch.isnan(d).any() else d.max().item()


def verdict(v, ok, bad):
    if v != v:  # NaN
        return "NaN in output"
    return ok if v <= TOL else bad


torch.manual_seed(1)
H = 64
ids_a = torch.randint(0, 256, (1, LA))
ids_b = torch.randint(0, 256, (1, LB))
hs_a = torch.randn(1, LA, H)
hs_b = torch.randn(1, LB, H)
ids_b2, hs_b2 = ids_b.clone(), hs_b.clone()
ids_b2[0, -1] = (ids_b2[0, -1] + 1) % 256
hs_b2[0, -1] += 1.0
pos_reset = torch.cat([torch.arange(LA), torch.arange(LB)]).unsqueeze(0)

lines = [f"python {platform.python_version()}  torch {torch.__version__}  transformers {transformers.__version__}",
         f"drafter class: {MODEL_FILE}", f"tolerance: {TOL}", ""]

for requested, label in [("flash_attention_2", "AS DEPLOYED: actor loaded with flash_attention_2 -> drafter swaps to eager"),
                         ("sdpa", "COMPARISON: default sdpa backend")]:
    model = build(requested)
    mask_path = ("transformers _update_causal_mask" if hasattr(model.model, "_update_causal_mask")
                 else "fastrl fallback (torch.triu boolean mask, qwen2_eagle.py:98-114)")
    alone = run(model, ids_b, hs_b)
    packed = run(model, torch.cat([ids_a, ids_b], 1), torch.cat([hs_a, hs_b], 1))
    packed_reset = run(model, torch.cat([ids_a, ids_b], 1), torch.cat([hs_a, hs_b], 1), position_ids=pos_reset)
    c0 = diff(run(model, ids_b, hs_b), alone)
    c1 = diff(alone[:-1], run(model, ids_b2, hs_b2)[:-1])
    c2 = diff(packed[LA:], alone)
    c2r = diff(packed[:LA], run(model, ids_a, hs_a))
    c3 = diff(packed_reset[LA:], alone)
    lines += [
        f"== {label}",
        f"   attention backend in use: {model.config._attn_implementation};  mask path: {mask_path}",
        f"   C0 control, identical inputs          max|diff| = {c0:.3e}  -> {verdict(c0, 'deterministic', 'NON-DETERMINISTIC')}",
        f"   C1 change last token, earlier outputs max|diff| = {c1:.3e}  -> "
        f"{verdict(c1, 'causal', 'NOT CAUSAL: earlier positions see a later token')}",
        f"   C2 B packed after A vs. B alone       max|diff| = {c2:.3e}  -> {verdict(c2, 'no cross-sample effect', 'cross-sample effect')}",
        f"   C2' A packed before B vs. A alone      max|diff| = {c2r:.3e}  -> "
        f"{verdict(c2r, 'earlier sample unaffected', 'earlier sample sees the LATER sample')}",
        f"   C3 C2 with positions reset            max|diff| = {c3:.3e}  -> "
        f"{verdict(c3, 'explained by positions only', 'attention crosses the sample boundary')}",
        "",
    ]
# Positive control: the stock Hugging Face Qwen2ForCausalLM, same size and inputs, eager attention.
# It must come out causal, which shows the harness can return that verdict.
from transformers import Qwen2ForCausalLM  # noqa: E402

torch.manual_seed(0)
ref_cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      use_cache=False, attn_implementation="eager")
ref = Qwen2ForCausalLM(ref_cfg).float().eval()


@torch.no_grad()
def run_ref(ids):
    return ref(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states[-1][0]


p1 = diff(run_ref(ids_b)[:-1], run_ref(ids_b2)[:-1])
lines += ["== POSITIVE CONTROL: stock transformers Qwen2ForCausalLM (eager)",
          f"   C1 change last token, earlier outputs max|diff| = {p1:.3e}  -> "
          f"{verdict(p1, 'causal (harness can detect a correct mask)', 'NOT CAUSAL (harness problem)')}", ""]
print("\n".join(lines))
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_drafter_masking_results.txt")
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nresults written to {out_path}")
