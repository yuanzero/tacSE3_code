"""Replay a saved NPZ with explicit unit assumptions; never import a device SDK."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core.estimator import estimate_rotation_from_shear
from .core.units import pixels_per_mm, shear_to_m_scale, DMV1_SDK_SHA256


def metric_fields(depth, shear, *, depth_unit, shear_unit, ppmm):
    if not np.isfinite(ppmm) or ppmm <= 0:
        raise ValueError('ppmm must be positive and finite')
    depth_scale = {'m': 1., 'mm': .001}[depth_unit]
    shear_scale = shear_to_m_scale(shear_unit, ppmm)
    depth = np.asarray(depth, dtype=float) * depth_scale
    shear = np.asarray(shear, dtype=float) * shear_scale
    if depth.ndim != 2 or depth.size == 0 or shear.shape != (*depth.shape, 2):
        raise ValueError('expected depth HxW and shear HxWx2')
    if not np.isfinite(depth).all() or not np.isfinite(shear).all():
        raise ValueError('nonfinite input')
    return depth, shear


def replay(depth, shear, *, depth_unit, shear_unit, ppmm, threshold_mm=.5, include_translation=True):
    if not np.isfinite(threshold_mm) or threshold_mm < 0:
        raise ValueError('contact threshold must be nonnegative and finite')
    depth, shear = metric_fields(depth, shear, depth_unit=depth_unit, shear_unit=shear_unit, ppmm=ppmm)
    mask = depth > threshold_mm * .001
    count = int(mask.sum())
    result = {'physical_accuracy_validated': False, 'actuator_commands_sent': 0,
              'assumptions': dict(depth_unit=depth_unit, shear_unit=shear_unit, ppmm=ppmm, threshold_mm=threshold_mm, include_translation=include_translation),
              'contact_pixels': count, 'image_width_mm': depth.shape[1] / ppmm}
    result['calibration'] = dict(status='nominal_unvalidated', mm_per_pixel=1. / ppmm,
        shear_to_m=shear_to_m_scale(shear_unit, ppmm),
        sdk_profile_sha256=DMV1_SDK_SHA256 if shear_unit == 'dmv1-code' else None,
        note='SDK profile must match capture SDK; spatial/depth/shear physical calibration is not established.')
    if count < 3:
        return dict(result, status='insufficient_contact')
    estimate = estimate_rotation_from_shear(depth, shear, ppmm=ppmm, contact_mask=mask,
                                           include_translation=include_translation, use_divfree_only=False)
    vector = np.asarray(estimate['omega_rad'], dtype=float)
    rank = int(estimate['rank'])
    flags = []
    if rank < (5 if include_translation else 3):flags.append('rank_deficient')
    if np.linalg.norm(vector) > .15:flags.append('large_rotation_outside_small_motion_regime')
    # Estimator residuals may be empty on rank deficiency, so do not report a false zero fit error.
    result.update(status='diagnostic_only', rotation_vector_rad=vector.tolist(),
                  rotation_norm_deg=float(np.degrees(np.linalg.norm(vector))), rank=rank,
                  singular_values=estimate['s'].tolist(), flags=flags)
    if include_translation:result['translation_xy_m'] = estimate['translation_m'].tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=Path)
    parser.add_argument('--depth-unit', choices=['m', 'mm'], required=True)
    parser.add_argument('--shear-unit', choices=['m', 'mm', 'pixel', 'dmv1-code'], required=True)
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument('--ppmm', type=float, help='Actual pixels per millimeter (not historical NormalFlow ppmm)')
    scale.add_argument('--mm-per-pixel', type=float, help='Nominal historical value: 0.0634; requires physical calibration')
    parser.add_argument('--threshold-mm', type=float, default=.5)
    parser.add_argument('--rotation-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mm_per_pixel is not None:
        args.ppmm = pixels_per_mm(args.mm_per_pixel)
    with np.load(args.sample, allow_pickle=False) as sample:
        result = replay(sample['depth'], sample['shear'], depth_unit=args.depth_unit,
                        shear_unit=args.shear_unit, ppmm=args.ppmm, threshold_mm=args.threshold_mm,
                        include_translation=not args.rotation_only)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
