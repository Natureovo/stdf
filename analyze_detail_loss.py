import argparse
import importlib.util
import os.path as op


def _load_detail_loss_module():
    module_path = op.join(op.dirname(__file__), 'utils', 'detail_loss.py')
    spec = importlib.util.spec_from_file_location('detail_loss_module', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze compressed image detail loss in gradient and frequency domains.'
    )
    parser.add_argument('--ref', required=True, help='Reference/original image path.')
    parser.add_argument('--cmp', required=True, help='Compressed image path.')
    parser.add_argument('--out', default='outputs/detail_loss', help='Output root directory.')
    parser.add_argument('--case-name', default=None, help='Optional output subfolder name.')
    parser.add_argument('--block', type=int, default=32, help='Block size for DCT statistics.')
    parser.add_argument('--threshold', type=float, default=0.55, help='Candidate mask threshold.')
    parser.add_argument('--save-full', action='store_true', help='Save detailed gradient/frequency maps and CSV.')
    return parser.parse_args()


def main():
    args = parse_args()
    detail_loss = _load_detail_loss_module()
    case_name = args.case_name
    if case_name is None:
        ref_name = op.splitext(op.basename(args.ref))[0]
        cmp_name = op.splitext(op.basename(args.cmp))[0]
        case_name = f'{ref_name}_vs_{cmp_name}'
    out_dir = op.join(args.out, case_name)

    result = detail_loss.analyze_image_files(
        args.ref,
        args.cmp,
        out_dir=out_dir,
        block_size=args.block,
        threshold=args.threshold,
        save_full=args.save_full,
    )
    report = result['report']
    print(f'analysis saved to: {out_dir}')
    print('detail_loss_mean: {:.4f}'.format(report['global']['detail_loss_mean']))
    print('candidate_area_ratio: {:.4f}'.format(report['global']['candidate_area_ratio']))
    print('candidate_region_count: {}'.format(report['regional']['candidate_region_count']))


if __name__ == '__main__':
    main()
