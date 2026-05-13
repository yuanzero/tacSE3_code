import numpy as np

def build_contact_from_daimon(depth, shear, ppmm, depth_in_mm=False,
                              shear_scale=1.0, contact_thresh_mm=0.5,
                              baseline_depth_mm=None, press_increases_depth=False):
    """
    Process raw depth and shear frames into metric units and a contact mask.
    """
    depth_arr = np.array(depth, dtype=np.float32)
    if not depth_in_mm:
        depth_mm = depth_arr * 1000.0
    else:
        depth_mm = depth_arr

    shear_arr = None
    if shear is not None:
        shear_arr = np.array(shear, dtype=np.float32) * float(shear_scale)
        if shear_arr.ndim != 3 or shear_arr.shape[2] != 2:
            raise ValueError("shear must have shape (H, W, 2)")
    
    # Calculate depth in meters (for later use, though function returns depth_mm primarily for thresholding)
    depth_m = depth_mm * 1e-3
    
    # Process shear to meters
    # shear is usually in pixels? The scale factor accounts for this to some extent.
    # The original code passes shear through.
    
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
         # Assuming shear_arr is already scaled appropriately or is raw
         # If ppmm is provided, we might want to scale shear from pixels?
         # The original implementation mainly just multiplied by shear_scale
         shear_out = shear_arr 
    else:
         shear_out = None

    return depth_m, shear_out, C
