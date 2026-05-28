import torch
import importlib.util
import os.path as op

from net_grdr import build_grdr


def _load_detail_guidance():
    module_path = op.join(op.dirname(__file__), 'utils', 'detail_guidance.py')
    spec = importlib.util.spec_from_file_location('detail_guidance_module', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_detail_guidance


def main():
    compute_detail_guidance = _load_detail_guidance()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, h, w = 2, 64, 64
    lq = torch.rand(b, 1, h, w, device=device)
    base = torch.rand(b, 1, h, w, device=device)
    gt = torch.rand(b, 1, h, w, device=device)

    guidance_maps = compute_detail_guidance(gt, base)
    guidance = guidance_maps['guidance']

    model = build_grdr({
        'in_nc': 1,
        'nf': 16,
        'cond_dim': 64,
        'rate_dim': 1,
        'num_steps': 20,
    }).to(device)
    rate_cond = torch.zeros(b, 1, device=device)
    loss = model.training_loss(lq, base, gt, guidance, rate_cond=rate_cond)
    loss.backward()
    refined = model.refine(lq, base, guidance, rate_cond=rate_cond, steps=5)

    print('device:', device)
    print('guidance shape:', tuple(guidance.shape))
    print('guidance range: [{:.4f}, {:.4f}]'.format(
        float(guidance.min().detach().cpu()),
        float(guidance.max().detach().cpu())
    ))
    print('loss:', float(loss.detach().cpu()))
    print('refined shape:', tuple(refined.shape))


if __name__ == '__main__':
    main()
