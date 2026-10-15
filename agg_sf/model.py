import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, c, shape):
        super().__init__()
        self.c = c
        if len(shape) != 3 or len(c['regions']) != shape[0]:
            raise ValueError('Use N,channels,bands,time features and one region index per channel')
        self.regions = c['regions']
        self.project = nn.Linear(shape[1], c['width'])
        self.functional = nn.Parameter(torch.eye(c['width']))
        self.layers = nn.ModuleList([nn.Linear(c['width'], c['width']) for _ in range(2)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(c['width']) for _ in range(2)])
        self.down = nn.ModuleList([nn.Linear(c['width'], c['bottleneck']) for _ in range(2)])
        self.up = nn.ModuleList([nn.Linear(c['bottleneck'], c['width']) for _ in range(2)])
        for layer in self.up:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for group in [self.down, self.up]:
            for parameter in group.parameters():
                parameter.requires_grad_(False)
        self.classifier = nn.Linear(c['width'], c['classes'])
        self.decoder = nn.Linear(c['width'], shape[1])
        self.register_buffer('anatomy', torch.tensor(c['region_adjacency'], dtype=torch.float32))
        if c['variant'] not in ['full', 'no-anatomy', 'no-reconstruction', 'no-band', 'no-neighbor', 'entropy-only', 'source-only']:
            raise ValueError('Unknown variant')

    def forward(self, batch):
        x = batch['x'].float()
        b, channels, bands, length = x.shape
        available = batch.get('available', torch.ones(b, channels, device=x.device)).bool()
        nodes, valid = [], []
        for region in range(5):
            ids = [i for i, r in enumerate(self.regions) if r == region]
            if ids:
                mask = available[:, ids]
                pooled = (x[:, ids].mean(-1) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                nodes.append(pooled)
                valid.append(mask.any(1))
            else:
                nodes.append(torch.zeros(b, bands, device=x.device))
                valid.append(torch.zeros(b, dtype=torch.bool, device=x.device))
        present = torch.stack(valid, 1)
        h = self.project(torch.stack(nodes, 1))
        pair = present[:, :, None] & present[:, None, :]
        functional = F.relu(h @ self.functional @ h.transpose(1, 2)).masked_fill(~pair, -1e4).softmax(-1) * pair
        alpha = 0 if self.c['variant'] == 'no-anatomy' else self.c['anatomy_weight']
        a = (alpha * self.anatomy[None] + (1 - alpha) * functional) * pair
        a = .5 * (a + a.transpose(1, 2)) + torch.diag_embed(present.float())
        degree = a.sum(-1).clamp_min(1e-6).rsqrt()
        a = degree[:, :, None] * a * degree[:, None, :]
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(a @ h))
            h = self.norms[i](h.transpose(1, 2)).transpose(1, 2)
            h = (h + self.up[i](F.relu(self.down[i](h)))) * present[..., None]
        representation = h.sum(1) / present.sum(1, keepdim=True).clamp_min(1)
        reconstruction = self.decoder(h)[:, self.regions, :, None].expand(-1, -1, -1, length)
        return dict(logits=self.classifier(representation), representation=representation,
                    reconstruction=reconstruction, region_available=present)

    def auxiliary(self, batch):
        x = batch['x'].float()
        mask = torch.rand_like(x) < self.c['cell_mask_rate']
        hidden = self(dict(batch, x=x.masked_fill(mask, 0)))
        reconstruction = F.mse_loss(hidden['reconstruction'][mask], x[mask]) if mask.any() else hidden['logits'].sum() * 0
        dropped = x.clone()
        band = int(torch.randint(x.shape[2], (1,)))
        dropped[:, :, band] = 0
        p = self(batch)['logits'].softmax(-1)
        q = self(dict(batch, x=dropped))['logits'].log_softmax(-1)
        consistency = F.kl_div(q, p.detach(), reduction='batchmean')
        if self.c['variant'] == 'no-reconstruction':
            reconstruction = reconstruction * 0
        if self.c['variant'] == 'no-band':
            consistency = consistency * 0
        return self.c['reconstruction_weight'] * reconstruction + self.c['band_weight'] * consistency

    def loss(self, batch, c, epoch):
        return F.cross_entropy(self(batch)['logits'], batch['y']) + self.auxiliary(batch)

    def target_loss(self, batch):
        result = self(batch)
        p = result['logits'].softmax(-1)
        confidence, pseudo = p.detach().max(-1)
        selected = confidence >= self.c['confidence_threshold']
        loss = F.cross_entropy(result['logits'][selected], pseudo[selected]) if selected.any() else p.sum() * 0
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1).mean()
        mean = p.mean(0)
        information = entropy + (mean * mean.clamp_min(1e-8).log()).sum()
        distances = torch.cdist(result['representation'].detach(), result['representation'].detach())
        distances.fill_diagonal_(torch.inf)
        k = min(self.c['neighbors'], len(p) - 1)
        neighbors = distances.topk(k, largest=False).indices
        neighbor_loss = (p[:, None] - p[neighbors]).square().mean() if k else p.sum() * 0
        if self.c['variant'] == 'no-neighbor':
            neighbor_loss = neighbor_loss * 0
        if self.c['variant'] == 'entropy-only':
            return entropy
        return loss + self.c['information_weight'] * information + self.auxiliary(batch) + self.c['neighbor_weight'] * neighbor_loss


def extra_metrics(d, c):
    return dict(region_availability=float(np.mean(d['region_available'])))
