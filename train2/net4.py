"""net4: card-embedding + self-attention + POINTER action heads for engine2.

Tabula rasa: contains no Dominion strategy knowledge of any kind. Every
parameter is shared across cards and kingdoms; what each card *means* is
learned entirely from self-play outcomes.

Interface contract (drop-in everywhere PolicyNetV3 was used):
    net = PolicyNetV4()
    logits, value = net(obs, mask)     # obs (B,345) float32, mask (B,106) bool
    aux = net.aux_out                  # (B,2) set by the last forward; the
                                       # trainer MAY use it (score-margin /
                                       # own-VP prediction heads). Rules-
                                       # derived targets only; never rewarded.

Action-space mapping (engine2):
    PLAY_BASE 0..17 -> PLAYABLE[i]      (card-parameterized -> pointer head)
    END_ACTION 18                       (global head)
    BUY_BASE 19..45 -> card 0..26       (pointer head)
    PASS 46                             (global head)
    GAIN_BASE 47..73 -> card 0..26      (pointer head)
    TRASH_BASE 74..100 -> card 0..26    (pointer head)
    STOP_TRASH 101, JESTER_TO_OPP 102, JESTER_TO_ME 103,
    HG_DUCHY 104, HG_ESTATES 105        (global head)

Pointer heads: for each card token (post-attention) a SHARED linear map
produces 4 logits: (play, buy, gain, trash) for THAT card. So "buy the
trasher when junk is flowing" is one learned function applied to every
card's contextual representation, instead of 27 independently-memorized
output slots. The 6 non-card actions come from a small global head on the
pooled board state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dominion.engine2 import (N_ACTIONS, N_CARDS, OBS_SIZE_V3, PLAYABLE,
                              PLAY_BASE, END_ACTION, BUY_BASE, PASS,
                              GAIN_BASE, TRASH_BASE, STOP_TRASH)

N_PLANES = 12
_CARD_BLOCK = N_PLANES * N_CARDS          # 324
N_GLOBALS = OBS_SIZE_V3 - _CARD_BLOCK     # 21
N_PLAYABLE = len(PLAYABLE)                # 18

assert _CARD_BLOCK + N_GLOBALS == OBS_SIZE_V3

# index map: which card does each PLAY action refer to
_PLAY_CARD = torch.tensor(PLAYABLE, dtype=torch.long)        # (18,)
# global (non-card) action ids, in a fixed order
_GLOBAL_ACTIONS = [END_ACTION, PASS, STOP_TRASH, 102, 103, 104, 105]


class _Block(nn.Module):
    """Pre-norm transformer encoder block over the 27 card tokens."""

    def __init__(self, dim, heads, ff_mult=2):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                                nn.Linear(dim * ff_mult, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class PolicyNetV4(nn.Module):
    obs_size = OBS_SIZE_V3

    def __init__(self, dim=96, heads=4, depth=3, hidden=384):
        super().__init__()
        self.card_emb = nn.Embedding(N_CARDS, dim)
        self.feat_proj = nn.Linear(N_PLANES, dim)
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(depth))
        self.card_norm = nn.LayerNorm(dim)
        self.global_proj = nn.Sequential(nn.Linear(N_GLOBALS, dim), nn.GELU())

        fused = dim * 2 + dim                       # mean ++ max ++ globals
        self.trunk = nn.Sequential(nn.Linear(fused, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())

        # pointer head: per-card-token -> (play, buy, gain, trash) logits.
        # Conditioned on the board summary via FiLM-style add so the same
        # card can score differently in different game phases.
        self.tok_cond = nn.Linear(hidden, dim)
        self.ptr = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, 4))
        # global head: the 7 non-card actions
        self.glob_head = nn.Linear(hidden, len(_GLOBAL_ACTIONS))
        self.v = nn.Linear(hidden, 1)
        # auxiliary heads (rules-derived prediction targets, NEVER reward):
        #   [0] tanh-scaled final score margin, [1] own final VP / 50
        self.aux = nn.Linear(hidden, 2)
        self.aux_out = None

        self.register_buffer("card_ids", torch.arange(N_CARDS),
                             persistent=False)
        self.register_buffer("play_card", _PLAY_CARD.clone(),
                             persistent=False)
        # commitment input: added to the token of the card this player is
        # committed to (exploration scaffolding, see selfplay4). Zero-init
        # so a warm-started net plays exactly as before until trained.
        self.commit_vec = nn.Parameter(torch.zeros(dim))

    def load_state_dict(self, sd, strict=True, **kw):
        if "commit_vec" not in sd:        # checkpoints from before commits
            sd = dict(sd)
            sd["commit_vec"] = torch.zeros_like(self.commit_vec)
        return super().load_state_dict(sd, strict, **kw)

    def forward(self, obs, mask, commit=None):
        """commit: optional (B,27) one-hot of the mover's committed card
        (all-zero rows = uncommitted). None = uncommitted everywhere."""
        B = obs.shape[0]
        feats = obs[:, :_CARD_BLOCK].reshape(B, N_PLANES, N_CARDS) \
                                    .transpose(1, 2)           # (B,27,12)
        glob = obs[:, _CARD_BLOCK:]                            # (B,21)

        tok = self.feat_proj(feats) + self.card_emb(self.card_ids)[None]
        if commit is not None:
            tok = tok + commit[:, :, None] * self.commit_vec
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.card_norm(tok)                              # (B,27,D)

        board = torch.cat([tok.mean(1), tok.amax(1)], -1)
        h = self.trunk(torch.cat([board, self.global_proj(glob)], -1))

        # per-card action logits, conditioned on global state
        ptok = tok + self.tok_cond(h)[:, None, :]              # (B,27,D)
        per_card = self.ptr(ptok)                              # (B,27,4)

        logits = obs.new_full((B, N_ACTIONS), 0.0)
        # play actions index into PLAYABLE's card ids
        logits[:, PLAY_BASE:PLAY_BASE + N_PLAYABLE] = \
            per_card[:, self.play_card, 0]
        logits[:, BUY_BASE:BUY_BASE + N_CARDS] = per_card[:, :, 1]
        logits[:, GAIN_BASE:GAIN_BASE + N_CARDS] = per_card[:, :, 2]
        logits[:, TRASH_BASE:TRASH_BASE + N_CARDS] = per_card[:, :, 3]
        g = self.glob_head(h)                                  # (B,7)
        for i, a in enumerate(_GLOBAL_ACTIONS):
            logits[:, a] = g[:, i]

        self.aux_out = self.aux(h)                             # (B,2)
        value = self.v(h).squeeze(-1)
        return logits.masked_fill(~mask, -1e9), value
