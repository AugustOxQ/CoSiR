"""Paper PercepT training settings; core's verbatim functions stay separate."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261018_percept_core'))
import pc_sept


def build_autoencoder(input_dim):
    """Paper's third 500-wide layer, mirrored around the 128-wide latent."""
    widths = [input_dim, 500, 500, 500, 2000, 128]

    def network(dims):
        layers = []
        for i, (left, right) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    return network(widths), network(widths[::-1])


def cosine_scheduler(optimizer, epochs):
    """Advance once after each epoch; reach 1e-5 at the phase maximum."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)


def pretrain_autoencoder(encoder, decoder, inputs, device, *, epochs=100):
    """Reconstruct clean inputs from Gaussian-corrupted latents in batches of 512."""
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=.001)
    scheduler = cosine_scheduler(optimizer, epochs)
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(inputs))
        total = 0.0
        for start in range(0, len(inputs), 512):
            batch = inputs[permutation[start:start + 512]].to(device)
            latent = encoder(batch)
            loss = F.mse_loss(decoder(latent + .1 * torch.randn_like(latent)), batch)
            assert torch.isfinite(loss), 'nonfinite pretraining loss'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(inputs))
        scheduler.step()
        print(f'Paper pretrain epoch={epoch} reconstruction={losses[-1]:.6f}', flush=True)
    return losses


@torch.no_grad()
def fit_assignments(encoder, centers, inputs, device):
    """Full-fit Q in frozen pair order, with bounded device memory."""
    encoder.eval()
    return torch.cat([pc_sept.soft_assignments(encoder(inputs[start:start + 512].to(device)),
                                               centers).cpu()
                      for start in range(0, len(inputs), 512)])


def train_dec_until_stable(encoder, decoder, centers, inputs, device, *,
                           max_epochs=200, stability_threshold=.001):
    """Refresh global P at each epoch start, then optimize shuffled fit minibatches."""
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                                 [centers], lr=.001)
    scheduler = cosine_scheduler(optimizer, max_epochs)
    history = []
    for epoch in range(1, max_epochs + 1):
        q_start = fit_assignments(encoder, centers, inputs, device)
        p = pc_sept.target_distribution(q_start)
        previous = q_start.argmax(dim=1)
        encoder.train()
        decoder.train()
        permutation = torch.randperm(len(inputs))
        totals = dict(total=0., kl=0., reconstruction=0.)
        lr = optimizer.param_groups[0]['lr']
        for start in range(0, len(inputs), 512):
            rows = permutation[start:start + 512]
            batch = inputs[rows].to(device)
            latent = encoder(batch)
            q = pc_sept.soft_assignments(latent, centers)
            kl = F.kl_div(q.log(), p[rows].to(device), reduction='batchmean')
            reconstruction = F.mse_loss(decoder(latent), batch)
            loss = kl + reconstruction
            assert torch.isfinite(loss), 'nonfinite DEC loss'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in (('total', loss), ('kl', kl), ('reconstruction', reconstruction)):
                totals[key] += value.item() * len(batch)
        current = fit_assignments(encoder, centers, inputs, device).argmax(dim=1)
        fraction = (current != previous).sum().item() / len(inputs)
        scheduler.step()
        history.append(dict(epoch=epoch, **{key: value / len(inputs) for key, value in totals.items()},
                            fraction_changed=fraction, lr=lr, next_lr=optimizer.param_groups[0]['lr']))
        print(f'Paper DEC epoch={epoch} total={history[-1]["total"]:.6f} '
              f'fraction_changed={fraction:.6f}', flush=True)
        if fraction < stability_threshold or epoch == max_epochs:
            reason = 'stability criterion' if fraction < stability_threshold else 'epoch ceiling'
            return history, reason, epoch


def stage1_config(config):
    """Replace core training metadata with the paper recipe, retaining input provenance."""
    config.update(encoder=[config['input_dim'], 500, 500, 500, 2000, 128],
                  decoder=[128, 2000, 500, 500, 500, config['input_dim']],
                  pretrain_batch_size=512, latent_noise_std=.1,
                  pretrain_target='clean input', pretrain_lr=.001, joint_lr=.001,
                  scheduler='cosine per epoch', eta_min=1e-5,
                  max_joint_steps=5 if config['smoke'] else 200,
                  joint_batch='shuffled fit minibatches; include tail', joint_batch_size=512,
                  joint_loss='KL(P||Q) batchmean + mean MSE', balance_weight=0,
                  stop_rule='post-epoch hard-label change fraction < threshold or epoch ceiling',
                  p_refresh_steps=None, p_refresh='all fit pairs at each epoch start',
                  pruning='one run; strict Algorithm 1 gamma=2 and A3 top 40',
                  topic_order='ascending original kept center index')
    config.pop('balance_log_clamp')
    return config


def train_mapper(inputs, targets, config, k):
    """Train paper's attention mapper with BCE and retain its final CPU weights."""
    import time
    import numpy as np

    seed, device = config['seed'], config['device']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mapper = pc_sept.AttentionPoolingMapper(d_model=config['input_dim'], n_topics=k).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=.001)
    scheduler = cosine_scheduler(optimizer, config['epochs'])
    target = torch.from_numpy(targets['fit']).to(device)
    history, epoch_seconds = [], []
    mapper.train()
    for epoch in range(1, config['epochs'] + 1):
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        permutation = torch.randperm(len(target), device=inputs['fit'].device)
        total = 0.0
        for start in range(0, len(target), 64):
            rows = permutation[start:start + 64]
            loss = F.binary_cross_entropy_with_logits(mapper(inputs['fit'][rows]), target[rows])
            # V4: a non-finite Stage 2 loss is not a crash. Training goes on, the weights end
            # non-finite, and run_stage2 saves the artifact marked nonfinite (the seed collapsed).
            # Stage 1 (pretraining and DEC above) keeps its loud asserts: R-COL needs finite centers.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        history.append(total / len(target))
        scheduler.step()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - tick)
        print(f'Paper {config["arm"]} epoch={epoch} mean_bce={history[-1]:.6f}', flush=True)
    mapper.eval()
    with torch.no_grad():
        dev_prob = mapper(inputs['dev']).sigmoid().cpu().numpy()
        eval_prob = mapper(inputs['eval']).sigmoid().cpu().numpy()
    state = {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()}
    return eval_prob, dev_prob, history, epoch_seconds, state


def stage2_config(config):
    """Paper patch-reader optimization; the selected cut supplies its topic width."""
    assert config['arm'] in ('P1', 'P4')
    config.update(lr=.001, batch='shuffled fit minibatches; include tail', batch_size=64,
                  scheduler='cosine per epoch', eta_min=1e-5,
                  targets='multi-hot union of own partition caption topics; selected cut')
    return config
