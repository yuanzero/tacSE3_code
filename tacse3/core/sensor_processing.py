import numpy as np
from .units import shear_to_m_scale

def build_contact_from_daimon(depth, shear, ppmm, depth_in_mm=True,
                              shear_scale=1.0, contact_thresh_mm=0.5,
                              baseline_depth_mm=None, press_increases_depth=False,
                              shear_unit='dmv1-code'):
    """
    Process raw depth and shear frames into metric units and a contact mask.
    Default units target the audited DMV1 SDK (see core.units); nominal only.
    For already metric input explicitly pass shear_unit='m'. shear_scale is a
    dimensionless gain applied after the unit conversion.
    """
    depth_arr = np.array(depth, dtype=np.float32)
    if not depth_in_mm:
        depth_mm = depth_arr * 1000.0
    else:
        depth_mm = depth_arr

    shear_arr = None
    if shear is not None:
        shear_arr = np.array(shear, dtype=np.float32) * shear_to_m_scale(shear_unit, ppmm) * float(shear_scale)
        if shear_arr.ndim != 3 or shear_arr.shape[2] != 2:
            raise ValueError("shear must have shape (H, W, 2)")
    
    # Calculate depth in meters (for later use, though function returns depth_mm primarily for thresholding)
    depth_m = depth_mm * 1e-3
    
    
    # Create contact mask
    if baseline_depth_mm is not None:
        try:
            baseline = np.array(baseline_depth_mm, dtype=np.float32)
            if press_increases_depth:
                C = depth_mm > (baseline + float(contact_thresh_mm))
            else:
                C = depth_mm < (baseline - float(contact_thresh_mm))
        except Exception:
            if press_increases_depth:
                C = depth_mm > float(contact_thresh_mm)
            else:
                C = depth_mm < float(contact_thresh_mm)
    else:
        if press_increases_depth:
            C = depth_mm > float(contact_thresh_mm)
        else:
            C = depth_mm < float(contact_thresh_mm)
    C = C.astype(bool)
    
    # Original function returned (depth_m, shear_m, C_mask)
    if shear_arr is not None:
         shear_out = shear_arr 
    else:
         shear_out = None

    return depth_m, shear_out, C
