import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import Model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/target')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    data = np.load(args.input, allow_pickle=False)
    x = torch.tensor(data['x'], dtype=torch.float32)
    if 'y' in data.files:
        raise ValueError('Adaptation input must omit labels')
    if 'group' not in data.files:
        raise ValueError('Unlabeled adaptation input requires subject group IDs')
    mean = x.mean((0, 3), keepdim=True)
    scale = x.std((0, 3), keepdim=True).clamp_min(1e-6)
    x = (x - mean) / scale
    model = Model(c, x.shape[1:])
    model.load_state_dict(torch.load(args.source, map_location='cpu', weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for group in [model.down, model.up, model.norms]:
        for parameter in group.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=c['learning_rate'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'settings.json').write_text(json.dumps(c, indent=2))
    history = []
    torch.manual_seed(c['seed'])
    for epoch in range(c['adaptation_epochs']):
        model.train()
        losses = []
        if c['variant'] == 'source-only':
            break
        for ids in torch.randperm(len(x)).split(c['batch_size']):
            batch = dict(x=x[ids])
            if 'available' in data.files:
                batch['available'] = torch.tensor(data['available'][ids])
            loss = model.target_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, unlabeled_loss=np.mean(losses)))
    torch.save(model.state_dict(), out / 'model.pt')
    np.savez(out / 'target_normalization.npz', mean=mean.numpy(), scale=scale.numpy())
    np.save(out / 'adaptation_groups.npy', np.unique(data['group']))
    pd.DataFrame(history).to_csv(out / 'adaptation.csv', index=False)


if __name__ == '__main__':
    main()
