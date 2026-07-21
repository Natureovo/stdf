import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize matched STDF/deterministic/ResShift reports across QPs.'
        )
    )
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def load_report(path):
    with open(path, 'r') as file_pointer:
        report = json.load(file_pointer)
    comparisons = report.get('comparisons', {})
    required = [
        'resshift_minus_base_psnr',
        'resshift_minus_deterministic_psnr',
    ]
    missing = [name for name in required if name not in comparisons]
    if missing:
        raise ValueError(
            f'{path} is missing matched comparisons: {missing}'
        )
    return report


def main():
    args = parse_args()
    rows = []
    for path in args.reports:
        report = load_report(path)
        comparisons = report['comparisons']
        versus_base = comparisons['resshift_minus_base_psnr']
        versus_deterministic = comparisons[
            'resshift_minus_deterministic_psnr'
        ]
        metrics = report['metrics']
        hf_delta = (
            metrics['resshift']['highfreq_mae'] -
            metrics['deterministic']['highfreq_mae']
        )
        passed = (
            versus_base['low'] > 0.0 and
            versus_deterministic['low'] > 0.0 and
            hf_delta <= 0.0
        )
        rows.append({
            'qp': float(report['qp']),
            'report': path,
            'resshift_minus_base': versus_base,
            'resshift_minus_deterministic': versus_deterministic,
            'highfreq_mae_delta_vs_deterministic': hf_delta,
            'pass': bool(passed),
        })
    rows.sort(key=lambda row: row['qp'])
    passing = [row for row in rows if row['pass']]
    selected = max(
        passing,
        key=lambda row: row['resshift_minus_deterministic']['mean'],
        default=None,
    )
    result = {
        'criterion': (
            'The 95% video-level PSNR confidence interval must be positive '
            'versus both STDF and the parameter-matched deterministic U-Net; '
            'high-frequency MAE must not be worse than the U-Net.'
        ),
        'rows': rows,
        'selected_qp': selected['qp'] if selected else None,
        'selected_report': selected['report'] if selected else None,
    }

    print('QP | diffusion-STDF (95% CI) | diffusion-U-Net (95% CI) | HF delta | gate')
    for row in rows:
        base = row['resshift_minus_base']
        deterministic = row['resshift_minus_deterministic']
        print(
            f"{row['qp']:g} | {base['mean']:+.6f} "
            f"[{base['low']:+.6f}, {base['high']:+.6f}] | "
            f"{deterministic['mean']:+.6f} "
            f"[{deterministic['low']:+.6f}, {deterministic['high']:+.6f}] | "
            f"{row['highfreq_mae_delta_vs_deterministic']:+.8f} | "
            f"{'PASS' if row['pass'] else 'STOP'}"
        )
    if selected is None:
        print('selected QP: none; diffusion has not passed the matched control.')
    else:
        print(f"selected QP: {selected['qp']:g} ({selected['report']})")

    if args.out:
        with open(args.out, 'w') as file_pointer:
            json.dump(result, file_pointer, indent=2)
        print(f'summary saved to {args.out}')


if __name__ == '__main__':
    main()
