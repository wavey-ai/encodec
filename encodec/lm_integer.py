# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

import os
import re
import typing as tp

import torch
from torch import nn

# ---- Integer-friendly LM (deterministic logits quantization) ----

class LMModelInt(nn.Module):
    """
    Same topology as the float LM in encodec.model.LMModel, but we quantize logits
    to a fixed step for extra determinism across architectures/BLAS backends.
    """
    def __init__(self, n_q: int, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        from .modules import transformer as m
        self.card = card
        self.n_q = n_q
        self.dim = dim

        # streaming transformer, kept in float64 for reproducibility
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs).to(torch.float64)

        # one embedding + one head per codebook
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim, dtype=torch.float64) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card, dtype=torch.float64) for _ in range(n_q)])

        # quantize logits onto a grid to stabilize results
        self.logit_step = 1.0 / 64.0

    @torch.no_grad()
    def forward(self, indices: torch.Tensor,
                states: tp.Optional[tp.List[torch.Tensor]] = None,
                offset: int = 0):
        """
        indices: [B, K, T] with K == runtime codebooks used (<= self.n_q)
        Returns:
            probas_or_counts: [B, card, K, T] on float64
            states, offset: streaming state
        """
        B, K, T = indices.shape
        # Sum embeddings for the K active codebooks only.
        x = sum([self.emb[k](indices[:, k]) for k in range(K)])  # [B, T, dim]
        out, states, offset = self.transformer(x, states, offset)  # [B, T, dim]

        # Project per active codebook
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, T, card]
        logits = logits.permute(0, 3, 1, 2).contiguous()  # [B, card, K, T]

        # integer-like last mile: quantize logits then softmax
        logits = torch.round(logits / self.logit_step) * self.logit_step
        probas = torch.softmax(logits, dim=1)  # still float64; AC builder accepts any nonneg vector
        return probas, states, offset


# --------- Checkpoint loading (robust to head-count mismatch) ---------

def _infer_ckpt_nq(state: tp.Dict[str, torch.Tensor]) -> int:
    """
    Count how many emb.* / linears.* heads exist in the checkpoint.
    """
    head_idxs = []
    pat = re.compile(r'^(emb|linears)\.(\d+)\.')
    for k in state.keys():
        m = pat.match(k)
        if m:
            head_idxs.append(int(m.group(2)))
    return (max(head_idxs) + 1) if head_idxs else 0


def _desired_nq_from_model(model) -> int:
    """
    Compute number of codebooks actually used for the MODEL'S CURRENT bandwidth.
    """
    # Fallbacks if anything is missing
    default = getattr(getattr(model, 'quantizer', None), 'n_q', 32)

    try:
        # How many quantizers will be used at current bandwidth?
        q = model.quantizer  # RVQ
        fr = model.frame_rate
        bw = getattr(model, 'bandwidth', None)
        if bw is None:
            # If caller didn't set bandwidth yet, we conservatively use the max
            return default
        return q.get_num_quantizers_for_bandwidth(fr, bw)
    except Exception:
        return default


def _checkpoint_name_for(model_name: str) -> str:
    """
    Use the *existing* float-LM checkpoint names (they match architecture)
    so you don't need new files.
    """
    # These are the same filenames used by EnCodec's float LM.
    mapping = {
        'encodec_24khz': 'encodec_lm_24khz-1608e3c0.th',
        'encodec_48khz': 'encodec_lm_48khz-7add9fc3.th',
    }
    if model_name not in mapping:
        raise RuntimeError(f"N[48;48;201;1632;2814to LM checkpoint mapping for model '{model_name}'.")
    return mapping[model_name]


def _load_state_dict_from_url_or_env(ckpt_name: str):
    """
    If ENCODEC_LM_PATH is set, read from that folder; otherwise use torch.hub URL.
    """
    root = os.environ.get('ENCODEC_LM_PATH', '').strip()
    if root:
        path = os.path.join(root, ckpt_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ENCODEC_LM_PATH set, but file not found: {path}")
        state = torch.load(path, map_location='cpu')
    else:
        from .utils import _get_checkpoint_url
        url = _get_checkpoint_url('https://dl.fbaipublicfiles.com/encodec/v0/', ckpt_name)
        state = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)  # type: ignore
    return state


def load_pretrained_integer_lm(model_or_name, device='cpu',
                               n_q: tp.Optional[int] = None,
                               card: tp.Optional[int] = None) -> LMModelInt:
    """
    Build LMModelInt sized to the number of codebooks you actually use,
    then load a compatible checkpoint (even if the checkpoint has fewer heads).

    Args:
        model_or_name: EncodecModel instance (preferred), or string model name.
        device: target device.
        n_q: override number of heads; if None, computed from model's bandwidth.
        card: override codebook size; if None, taken from model.quantizer.bins.

    Returns:
        LMModelInt in eval mode (float64 params), on `device`.
    """
    if hasattr(model_or_name, 'name'):
        model_name = model_or_name.name
        if n_q is None:
            n_q = _desired_nq_from_model(model_or_name)
        if card is None:
            try:
                card = model_or_name.quantizer.bins
            except Exception:
                card = 1024
    else:
        # string path
        model_name = str(model_or_name)
        n_q = n_q or int(os.getenv('ENCODEC_LM_NQ', '32'))
        card = card or 1024

    # safety
    if n_q is None:
        n_q = 32
    if card is None:
        card = 1024

    # Build the model skeleton with the *desired* number of heads.
    lm = LMModelInt(n_q=n_q, card=card, num_layers=5, dim=200,
                    past_context=int(3.5 * getattr(getattr(model_or_name, 'frame_rate', 50), '__int__', lambda: 50)()))
    lm = lm.to(device=device, dtype=torch.float64)

    # Load float-LM checkpoint (layout-compatible)
    ckpt_name = _checkpoint_name_for(model_name)
    state = _load_state_dict_from_url_or_env(ckpt_name)
    ckpt_nq = _infer_ckpt_nq(state)

    # If checkpoint has fewer heads, we will load what exists, then clone the last head.
    # If it has more, we will load a slice of the heads.
    # Always load with strict=False so missing/extra per-head params are okay.
    missing, unexpected = lm.load_state_dict(state, strict=False)

    # If we asked for more heads than the checkpoint provides, synthesize the extra heads
    if ckpt_nq and n_q > ckpt_nq:
        with torch.no_grad():
            # pick a source head to clone (last available)
            src = ckpt_nq - 1
            for k in range(ckpt_nq, n_q):
                lm.emb[k].weight.copy_(lm.emb[src].weight)
                lm.linears[k].weight.copy_(lm.linears[src].weight)
                lm.linears[k].bias.copy_(lm.linears[src].bias)

    # Put in eval & float64
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
        p.data = p.data.to(dtype=torch.float64, device=device)

    # Helpful log (printed only if env set)
    if os.getenv('ENCODEC_LM_VERBOSE', ''):
        print(f"[lm_integer] model={model_name} desired_n_q={n_q} ckpt_heads={ckpt_nq} "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    return lm

