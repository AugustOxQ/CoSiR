"""Stage 2 trainer: chunked evaluation, fp16 residency and the paper optimisation (CPU, tiny sizes)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import po_percept as po

N, T, D, K = 150, 5, 8, 3


def tiny():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, T, D, generator=g).half()                      # fp16-rounded values
    y = (torch.rand(N, K, generator=g) > .6).float().numpy()
    config = dict(seed=3, device='cpu', epochs=3, arm='L1', input_dim=D)
    return x, y, config


def test_chunked_evaluation_equals_full_batch():
    x, _, _ = tiny()
    torch.manual_seed(1)
    mapper = po.pc_sept.AttentionPoolingMapper(d_model=D, n_topics=K).eval()
    for p in mapper.parameters():
        p.data = p.data * 5
    with torch.no_grad():
        full = mapper(x.float()).sigmoid().numpy()
    for chunk in (1, 7, 64, 150, 512):
        got = po.predict(mapper, lambda a, b: x[a:b].numpy(), N, 'cpu', chunk=chunk)
        assert got.dtype == np.float32 and np.allclose(got, full, atol=1e-6, rtol=0)


def test_fp16_inputs_give_the_same_weights_as_float32_holding_the_same_values():
    x, y, config = tiny()
    _, h16, _, s16 = po.train_mapper(x, y, config, K)
    _, h32, _, s32 = po.train_mapper(x.float(), y, config, K)
    assert h16 == h32
    for key in s16:
        assert torch.equal(s16[key], s32[key])
        assert s16[key].dtype == torch.float32


def test_trainer_matches_the_claim_test_trainer():
    x, y, config = tiny()
    _, history, _, state = po.train_mapper(x, y, config, K)
    ref_config = dict(config, arm='P1')
    inputs = dict(fit=x.float(), dev=x.float()[:10], eval=x.float()[:10])
    out = po.ct_paper_sept.train_mapper(inputs, dict(fit=y), ref_config, K)
    ref_eval, ref_dev, ref_history, _, ref_state = out
    assert history == ref_history
    for key in state:
        assert torch.equal(state[key], ref_state[key])
