import argparse
import json
import math
import os
import re
from collections import defaultdict


PIXEL_METRICS = (
    'rgb_psnr',
    'y_psnr',
    'chroma_psnr',
    'ssim',
    'highfreq_mae',
    'gradient_mae',
)
PERCEPTUAL_METRICS = ('lpips', 'dists')
COMPARISONS = {
    'full_frame_expert': (
        'diffusion_full_frame',
        'deterministic_full_frame',
    ),
    'need_only': (
        'diffusion_same_region',
        'deterministic',
    ),
    'final_route': (
        'diffusion_confidence_routed',
        'deterministic',
    ),
    'confidence_only_route': (
        'diffusion_confidence_only',
        'deterministic',
    ),
    'route_protection': (
        'diffusion_confidence_routed',
        'diffusion_same_region',
    ),
    'location_vs_matched_need': (
        'diffusion_confidence_routed',
        'diffusion_matched_mass_need',
    ),
    'location_vs_shifted': (
        'diffusion_confidence_routed',
        'diffusion_shifted_joint',
    ),
    'location_vs_multi_shift': (
        'diffusion_confidence_routed',
        'diffusion_multi_shift_control',
    ),
    'location_vs_block_permutation': (
        'diffusion_confidence_routed',
        'diffusion_block_permutation_control',
    ),
    'confidence_location_vs_multi_shift': (
        'diffusion_confidence_only',
        'diffusion_confidence_multi_shift_control',
    ),
    'confidence_location_vs_block_permutation': (
        'diffusion_confidence_only',
        'diffusion_confidence_block_permutation_control',
    ),
}
PROTOCOL_KEYS = (
    'target_mode',
    'diffusion_candidates',
    'diffusion_noise_mode',
    'seed',
    'eval_crop_size',
    'model_tile_size',
    'model_tile_overlap',
    'location_shift_controls',
    'location_permutation_controls',
    'location_block_size',
    'confidence_location_controls',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Pool routed diffusion reports without treating correlated QPs '
            'from one video as independent primary samples.'
        )
    )
    parser.add_argument('reports', nargs='+')
    parser.add_argument(
        '--output',
        default='outputs/routed_causal_combined.json',
    )
    parser.add_argument('--psnr_margin', type=float, default=0.02)
    parser.add_argument('--ssim_margin', type=float, default=0.002)
    parser.add_argument('--temporal_margin', type=float, default=0.0001)
    return parser.parse_args()


def confidence_interval(values):
    values = [float(value) for value in values]
    if not values:
        return {
            'mean': 0.0,
            'low': 0.0,
            'high': 0.0,
            'n': 0,
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {
            'mean': mean,
            'low': mean,
            'high': mean,
            'n': 1,
        }
    variance = sum(
        (value - mean) ** 2 for value in values
    ) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {
        'mean': mean,
        'low': mean - half_width,
        'high': mean + half_width,
        'n': len(values),
    }


def average(values):
    return sum(values) / len(values)


def base_video_name(video_name):
    return re.sub(
        r'_QP[-+]?\d+(?:\.\d+)?$',
        '',
        str(video_name),
    )


def comparison_delta(group, left, right, section, metric):
    section_values = group.get(section, {})
    if left not in section_values or right not in section_values:
        return None
    left_values = section_values[left]
    right_values = section_values[right]
    if metric not in left_values or metric not in right_values:
        return None
    return float(left_values[metric] - right_values[metric])


def load_reports(paths):
    loaded = []
    reference_protocol = None
    reference_checkpoints = None
    seen_video_splits = {}
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file_pointer:
            report = json.load(file_pointer)
        protocol = report.get('protocol', {})
        protocol_signature = {
            key: protocol.get(key) for key in PROTOCOL_KEYS
        }
        if reference_protocol is None:
            reference_protocol = protocol_signature
            reference_checkpoints = report.get('checkpoints', {})
        elif protocol_signature != reference_protocol:
            raise ValueError(
                'Incompatible protocols in {}: {} != {}.'.format(
                    path,
                    protocol_signature,
                    reference_protocol,
                )
            )
        elif report.get('checkpoints', {}) != reference_checkpoints:
            raise ValueError(
                'Checkpoint mismatch in {}.'.format(path)
            )

        split = str(protocol.get('split', 'unknown'))
        groups = report.get('by_video_qp', [])
        if not groups:
            raise ValueError(
                '{} does not contain by_video_qp records.'.format(path)
            )
        source = '{}:{}'.format(split, os.path.basename(path))
        for group in groups:
            video = base_video_name(group['video'])
            previous_split = seen_video_splits.get(video)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    'Video {} appears in both {} and {}; reports are not '
                    'independent.'.format(video, previous_split, split)
                )
            seen_video_splits[video] = split
        loaded.append({
            'path': path,
            'source': source,
            'split': split,
            'report': report,
        })
    return loaded, reference_protocol, reference_checkpoints


def collect_deltas(loaded_reports):
    collected = {
        comparison: {
            'pixel': {
                metric: [] for metric in PIXEL_METRICS
            },
            'perceptual': {
                metric: [] for metric in PERCEPTUAL_METRICS
            },
            'temporal': {'temporal_error': []},
        }
        for comparison in COMPARISONS
    }
    metadata = {}
    report_groups = defaultdict(set)
    for loaded in loaded_reports:
        source = loaded['source']
        for group in loaded['report']['by_video_qp']:
            video = base_video_name(group['video'])
            qp = int(group['qp'])
            video_key = '{}:{}'.format(loaded['split'], video)
            group_key = '{}:QP{}'.format(video_key, qp)
            metadata[group_key] = {
                'source': source,
                'split': loaded['split'],
                'video': video,
                'video_key': video_key,
                'qp': qp,
            }
            report_groups[source].add(group_key)
            for comparison, (left, right) in COMPARISONS.items():
                for metric in PIXEL_METRICS:
                    value = comparison_delta(
                        group,
                        left,
                        right,
                        'methods',
                        metric,
                    )
                    if value is not None:
                        collected[comparison]['pixel'][metric].append(
                            (group_key, value)
                        )
                for metric in PERCEPTUAL_METRICS:
                    value = comparison_delta(
                        group,
                        left,
                        right,
                        'perceptual',
                        metric,
                    )
                    if value is not None:
                        collected[comparison][
                            'perceptual'
                        ][metric].append((group_key, value))
                value = comparison_delta(
                    group,
                    left,
                    right,
                    'temporal',
                    'temporal_error',
                )
                if value is not None:
                    collected[comparison][
                        'temporal'
                    ]['temporal_error'].append((group_key, value))
    return collected, metadata, report_groups


def aggregate_entries(entries, metadata, unit, selector=None):
    grouped = defaultdict(list)
    for group_key, value in entries:
        item = metadata[group_key]
        if selector is not None and not selector(item):
            continue
        if unit == 'video':
            key = item['video_key']
        elif unit == 'video_qp':
            key = group_key
        else:
            raise ValueError('Unsupported aggregation unit: {}.'.format(unit))
        grouped[key].append(value)
    return confidence_interval(
        average(values) for values in grouped.values()
    )


def summarize_comparison(values, metadata, unit, selector=None):
    return {
        section: {
            metric: aggregate_entries(
                entries,
                metadata,
                unit,
                selector=selector,
            )
            for metric, entries in metrics.items()
        }
        for section, metrics in values.items()
    }


def evidence_status(interval, lower_is_better=True):
    if interval['n'] == 0:
        return 'NOT_RUN'
    if lower_is_better:
        if interval['high'] <= 0.0:
            return 'PASS'
        if interval['low'] > 0.0:
            return 'STOP'
    else:
        if interval['low'] >= 0.0:
            return 'PASS'
        if interval['high'] < 0.0:
            return 'STOP'
    return 'INCONCLUSIVE'


def noninferiority_status(interval, margin, lower_is_better=False):
    if interval['n'] == 0:
        return 'NOT_RUN'
    if lower_is_better:
        if interval['high'] <= float(margin):
            return 'PASS'
        if interval['low'] > float(margin):
            return 'STOP'
    else:
        if interval['low'] >= -float(margin):
            return 'PASS'
        if interval['high'] < -float(margin):
            return 'STOP'
    return 'INCONCLUSIVE'


def combine_statuses(*statuses):
    active = [status for status in statuses if status != 'NOT_RUN']
    if not active:
        return 'NOT_RUN'
    if 'STOP' in active:
        return 'STOP'
    if all(status == 'PASS' for status in active):
        return 'PASS'
    return 'INCONCLUSIVE'


def perceptual_noninferiority_gate(summary, args):
    return combine_statuses(
        noninferiority_status(
            summary['pixel']['rgb_psnr'],
            args.psnr_margin,
        ),
        noninferiority_status(
            summary['pixel']['ssim'],
            args.ssim_margin,
        ),
        evidence_status(summary['perceptual']['lpips']),
        evidence_status(summary['perceptual']['dists']),
        noninferiority_status(
            summary['temporal']['temporal_error'],
            args.temporal_margin,
            lower_is_better=True,
        ),
    )


def route_protection_gate(summary):
    return combine_statuses(
        evidence_status(
            summary['pixel']['rgb_psnr'],
            lower_is_better=False,
        ),
        evidence_status(
            summary['pixel']['ssim'],
            lower_is_better=False,
        ),
        evidence_status(summary['pixel']['highfreq_mae']),
        evidence_status(summary['pixel']['gradient_mae']),
    )


def format_interval(interval, digits=6):
    template = (
        '{:+.' + str(digits) + 'f} '
        '[{:+.' + str(digits) + 'f}, {:+.' + str(digits) + 'f}] '
        '(n={})'
    )
    return template.format(
        interval['mean'],
        interval['low'],
        interval['high'],
        interval['n'],
    )


def main():
    args = parse_args()
    for name in ('psnr_margin', 'ssim_margin', 'temporal_margin'):
        if getattr(args, name) < 0:
            raise ValueError('--{} must be non-negative.'.format(name))

    loaded, protocol, checkpoints = load_reports(args.reports)
    collected, metadata, report_groups = collect_deltas(loaded)
    qps = sorted({item['qp'] for item in metadata.values()})

    comparisons = {}
    for comparison, values in collected.items():
        video_summary = summarize_comparison(
            values,
            metadata,
            'video',
        )
        video_qp_summary = summarize_comparison(
            values,
            metadata,
            'video_qp',
        )
        by_qp = {
            str(qp): summarize_comparison(
                values,
                metadata,
                'video',
                selector=lambda item, target=qp: item['qp'] == target,
            )
            for qp in qps
        }
        by_report = {
            loaded_report['source']: summarize_comparison(
                values,
                metadata,
                'video',
                selector=(
                    lambda item, target=loaded_report['source']:
                    item['source'] == target
                ),
            )
            for loaded_report in loaded
        }
        comparisons[comparison] = {
            'primary_video_level': video_summary,
            'secondary_video_qp_level': video_qp_summary,
            'by_qp_video_level': by_qp,
            'by_report_video_level': by_report,
        }

    primary = {
        name: values['primary_video_level']
        for name, values in comparisons.items()
    }
    gates = {
        'full_frame_perceptual': combine_statuses(
            evidence_status(
                primary['full_frame_expert']['perceptual']['lpips']
            ),
            evidence_status(
                primary['full_frame_expert']['perceptual']['dists']
            ),
        ),
        'final_route': perceptual_noninferiority_gate(
            primary['final_route'],
            args,
        ),
        'confidence_only_route': perceptual_noninferiority_gate(
            primary['confidence_only_route'],
            args,
        ),
        'route_protection': route_protection_gate(
            primary['route_protection']
        ),
        'location_vs_matched_need': perceptual_noninferiority_gate(
            primary['location_vs_matched_need'],
            args,
        ),
        'location_vs_shifted': perceptual_noninferiority_gate(
            primary['location_vs_shifted'],
            args,
        ),
        'location_vs_multi_shift': perceptual_noninferiority_gate(
            primary['location_vs_multi_shift'],
            args,
        ),
        'location_vs_block_permutation': (
            perceptual_noninferiority_gate(
                primary['location_vs_block_permutation'],
                args,
            )
        ),
        'confidence_location_vs_multi_shift': (
            perceptual_noninferiority_gate(
                primary['confidence_location_vs_multi_shift'],
                args,
            )
        ),
        'confidence_location_vs_block_permutation': (
            perceptual_noninferiority_gate(
                primary['confidence_location_vs_block_permutation'],
                args,
            )
        ),
    }
    new_location_statuses = (
        gates['location_vs_multi_shift'],
        gates['location_vs_block_permutation'],
    )
    if all(status == 'NOT_RUN' for status in new_location_statuses):
        location_protocol = 'legacy_single_shift'
        gates['location'] = combine_statuses(
            gates['location_vs_matched_need'],
            gates['location_vs_shifted'],
        )
    else:
        location_protocol = 'multi_placebo_primary'
        gates['location'] = combine_statuses(
            gates['location_vs_matched_need'],
            gates['location_vs_multi_shift'],
            gates['location_vs_block_permutation'],
        )
    gates['causal'] = combine_statuses(
        gates['full_frame_perceptual'],
        gates['final_route'],
        gates['route_protection'],
        gates['location'],
    )
    confidence_location_statuses = (
        gates['confidence_location_vs_multi_shift'],
        gates['confidence_location_vs_block_permutation'],
    )
    if all(
            status == 'NOT_RUN'
            for status in confidence_location_statuses):
        gates['confidence_location'] = 'NOT_RUN'
    elif any(
            status == 'NOT_RUN'
            for status in confidence_location_statuses):
        gates['confidence_location'] = 'INCONCLUSIVE'
    else:
        gates['confidence_location'] = combine_statuses(
            *confidence_location_statuses
        )
    if gates['confidence_location'] == 'NOT_RUN':
        gates['confidence_causal'] = 'INCONCLUSIVE'
    else:
        gates['confidence_causal'] = combine_statuses(
            gates['full_frame_perceptual'],
            gates['confidence_only_route'],
            gates['confidence_location'],
        )

    videos = sorted({item['video_key'] for item in metadata.values()})
    result = {
        'protocol': {
            'primary_independent_unit': 'video',
            'secondary_unit': 'video_qp',
            'reports': args.reports,
            'videos': len(videos),
            'video_qp_groups': len(metadata),
            'qps': qps,
            'source_group_counts': {
                source: len(groups)
                for source, groups in report_groups.items()
            },
            'model_protocol': protocol,
            'location_gate_protocol': location_protocol,
            'noninferiority_margins': {
                'rgb_psnr': args.psnr_margin,
                'ssim': args.ssim_margin,
                'temporal_error': args.temporal_margin,
            },
        },
        'checkpoints': checkpoints,
        'comparisons': comparisons,
        'gates': gates,
    }

    title = (
        'Combined routed causal analysis'
        if len(loaded) > 1
        else 'Routed causal analysis'
    )
    print('========== {} =========='.format(title))
    print(
        'reports/videos/video-QP groups: {}/{}/{}'.format(
            len(loaded),
            len(videos),
            len(metadata),
        )
    )
    print('QP values: {}'.format('/'.join(map(str, qps))))
    print(
        'primary CI unit: video; secondary diagnostic unit: video-QP'
    )
    for comparison in COMPARISONS:
        summary = primary[comparison]
        print('\n-- {} --'.format(comparison))
        print(
            'RGB PSNR: {}'.format(
                format_interval(summary['pixel']['rgb_psnr'])
            )
        )
        print(
            'SSIM: {}'.format(
                format_interval(summary['pixel']['ssim'])
            )
        )
        print(
            'HF/gradient MAE: {} / {}'.format(
                format_interval(
                    summary['pixel']['highfreq_mae'],
                    digits=8,
                ),
                format_interval(
                    summary['pixel']['gradient_mae'],
                    digits=8,
                ),
            )
        )
        print(
            'LPIPS/DISTS: {} / {}'.format(
                format_interval(
                    summary['perceptual']['lpips'],
                    digits=8,
                ),
                format_interval(
                    summary['perceptual']['dists'],
                    digits=8,
                ),
            )
        )
        print(
            'temporal: {}'.format(
                format_interval(
                    summary['temporal']['temporal_error'],
                    digits=8,
                )
            )
        )
    print('\n-- Primary video-level gates --')
    for name, status in gates.items():
        print('{}: {}'.format(name, status))

    if args.output:
        output_directory = os.path.dirname(args.output)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as file_pointer:
            json.dump(result, file_pointer, indent=2)
        print('report saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
