import torch

from net_grdr import build_grdr


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_grdr({
        'in_nc': 1,
        'nf': 16,
        'cond_dim': 64,
        'rate_dim': 1,
        'num_steps': 20,
    }).to(device)

    b, h, w = 2, 64, 64
    lq = torch.rand(b, 1, h, w, device=device)
    base = torch.rand(b, 1, h, w, device=device)
    gt = torch.rand(b, 1, h, w, device=device)
    guidance = torch.rand(b, 1, h, w, device=device)
    rate_cond = torch.zeros(b, 1, device=device)

    loss = model.training_loss(lq, base, gt, guidance, rate_cond=rate_cond)
    loss.backward()
    refined = model.refine(lq, base, guidance, rate_cond=rate_cond, steps=5)

    print('device:', device)
    print('loss:', float(loss.detach().cpu()))
    print('refined shape:', tuple(refined.shape))
    print('refined range: [{:.4f}, {:.4f}]'.format(
        float(refined.min().detach().cpu()),
        float(refined.max().detach().cpu())
    ))


if __name__ == '__main__':
    main()
