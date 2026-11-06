"""Nominal source-derived units; these constants are not per-sensor calibration."""
import math

# Historical name ppmm in NormalFlow is misleading: its code multiplies by
# 0.0634 / 1000 to obtain meters. Our estimator uses actual pixels/mm.
NOMINAL_MM_PER_PIXEL = 0.0634
DEFAULT_PPMM = 1.0 / NOMINAL_MM_PER_PIXEL
DMV1_SDK_SHA256 = 'd8ad98da0f37b129a1e5fcee20d8a6730e4523c5353b55dc45adc11346acda17'
# Verified numerical chain for this SDK: flow * mmpp * 100, then * 0.05.
# Nominal displacement only: mmpp and tracker accuracy still need calibration.
DMV1_CODE_SHEAR_TO_M = 0.001 / 5.0

def pixels_per_mm(mm_per_pixel):
    value = float(mm_per_pixel)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('mm_per_pixel must be positive and finite')
    return 1.0 / value

def shear_to_m_scale(unit, ppmm):
    if not math.isfinite(ppmm) or ppmm <= 0:
        raise ValueError('ppmm must be positive and finite')
    return {'m': 1., 'mm': .001, 'pixel': .001 / ppmm,
            'dmv1-code': DMV1_CODE_SHEAR_TO_M}[unit]
