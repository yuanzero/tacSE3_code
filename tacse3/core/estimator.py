# Copied and adapted from realtime_daimon_rot/estimate_rot_from_shear.py

import numpy as np
import scipy.fft
from .units import DEFAULT_PPMM

def pixel_xy_grid(shape, ppmm):
    if not np.isfinite(ppmm) or ppmm <= 0:
        raise ValueError('ppmm must be positive and finite')
    H, W = shape
    xx = np.arange(W) - (W / 2.0) + 0.5
    yy = np.arange(H) - (H / 2.0) + 0.5
    xv, yv = np.meshgrid(xx, yy)
    scale = (1.0 / ppmm) * 1e-3
    # X corresponds to column index (width), Y to row index (height)
    X = xv * scale
    Y = yv * scale
    return X, Y


def helmholtz_decompose(U, V):
    """
    Decompose vector field (U, V) into divergence-free and curl-free components 
    using Helmholtz decomposition in Fourier domain.
    """
    import numpy.fft as fft

    H, W = U.shape
    # Create frequency grids (assuming periodic boundary conditions which FFT implies)
    # Note: For non-periodic data this is an approximation, but fast.
    ky = fft.fftfreq(H) * 2.0 * np.pi
    kx = fft.fftfreq(W) * 2.0 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    # Denominator k^2 for Poisson equation inversion
    denom = kx_grid ** 2 + ky_grid ** 2
    denom[0, 0] = 1.0 # Handle DC component (singularity) by ignoring it

    U_hat = fft.fft2(U)
    V_hat = fft.fft2(V)

    # Divergence field in Fourier domain: i(kx*U + ky*V)
    div_hat = 1j * (kx_grid * U_hat + ky_grid * V_hat)
    # Scalar potential phi_hat = - div_hat / k^2
    phi_hat = -div_hat / denom

    # Curl (vorticity) field in Fourier domain: i(kx*V - ky*U) (scalar in 2D)
    # Technically it's the z-component of curl
    curl_hat = 1j * (kx_grid * V_hat - ky_grid * U_hat)
    # Stream function psi_hat = - curl_hat / k^2
    psi_hat = -curl_hat / denom

    # Reconstruct Curl-free part (Gradient of phi): grad(phi)
    # U_cf = d(phi)/dx -> i*kx*phi
    # V_cf = d(phi)/dy -> i*ky*phi
    # (Note: signs and i factors depend on definition, but standard is i*k for derivative)
    # Actually, if div(F) = -Laplacian(phi), then F_irrot = -grad(phi). 
    # Let's trust the logic from original file:
    # U_curlfree = ifft(1j * kx * phi_hat)
    # V_curlfree = ifft(1j * ky * phi_hat)
    
    # But wait, original code has:
    # U_curlfree = np.real(fft.ifft2(1j * kx_grid * phi_hat))
    # This implies F_irrot = grad(phi) where phi solves Laplacian(phi) = div(F)
    # Wait, div(grad(phi)) = Lap(phi). If Lap(phi) = div(F), then grad(phi) matches divergence. Correct.
    
    U_curlfree = np.real(fft.ifft2(1j * kx_grid * phi_hat))
    V_curlfree = np.real(fft.ifft2(1j * ky_grid * phi_hat))

    # Reconstruct Divergence-free part (Curl of psi): curl(psi*k)
    # U_df = d(psi)/dy  (or -d(psi)/dy)
    # V_df = -d(psi)/dx (or d(psi)/dx)
    # Original code:
    # U_divfree = ifft(-1j * ky * psi_hat)
    # V_divfree = ifft(1j * kx * psi_hat)
    # This corresponds to curl(psi) = [dpsi/dy, -dpsi/dx] ??
    # check div: d(dpsi/dy)/dx + d(-dpsi/dx)/dy = d2psi/dxdy - d2psi/dydx = 0. Div free. Correct.
    
    U_divfree = np.real(fft.ifft2(-1j * ky_grid * psi_hat))
    V_divfree = np.real(fft.ifft2(1j * kx_grid * psi_hat))

    return U_divfree, V_divfree, U_curlfree, V_curlfree


def estimate_rotation_from_shear(depth_m, shear, ppmm=DEFAULT_PPMM, contact_mask=None,
                                 shear_scale=1.0, include_translation=True, rcond=None,
                                 use_divfree_only=True):
    """
    Estimate rotation vector omega from dense shear flow and depth map.
    
    Model: shear velocity v ≈ ω × r + t
    where r = (x, y, z) is the surface point.
    
    Args:
        depth_m: (H, W) depth map in meters
        shear: (H, W, 2) shear flow (u, v)
        ppmm: pixels per millimeter (calibration)
        contact_mask: (H, W) boolean mask of valid points
        shear_scale: scaling factor for shear values
        include_translation: if True, jointly estimate translation t
        use_divfree_only: if True, preprocess shear with Helmholtz decomp
        
    Returns:
        dict: {'omega_rad': [wx, wy, wz], 'translation_m': [tx, ty], 'residuals': ...}
    """
    if shear is None:
        raise ValueError("shear must be provided")

    depth_arr = np.array(depth_m, dtype=np.float32)
    H, W = depth_arr.shape
    
    # Scale shear
    # Shear is typically (u, v). If it's displacement, it has units of length.
    # The original code takes shear scale and just applies it.
    shear_arr = np.array(shear, dtype=np.float32) * float(shear_scale)
    if shear_arr.shape[:2] != (H, W):
        raise ValueError("shear must have same spatial shape as depth")

    if contact_mask is None:
        mag = np.linalg.norm(shear_arr, axis=2)
        contact_mask = mag > 1e-6
    contact_mask = contact_mask.astype(bool)

    # 1. Coordinate Grid
    X, Y = pixel_xy_grid((H, W), ppmm)
    Z = depth_arr

    # 2. Extract valid points
    yy, xx = np.nonzero(contact_mask)
    if len(xx) == 0:
         return {
            'omega_rad': np.zeros(3), 
            'omega_deg': np.zeros(3), 
            'translation_m': np.zeros(2) if include_translation else None,
            'residuals': float('inf'),
            'num_points': 0
        }

    # 3. Optional Helmholtz Decomposition
    #    Remove "squeeze" (divergence) component, keep "twist/shear" (curl/solenoidal)
    if use_divfree_only:
        U_full = shear_arr[:, :, 0]
        V_full = shear_arr[:, :, 1]
        
        # We perform decomp on the FULL frame (with zeros where mask is False?)
        # Or should we fill non-contact areas?
        # Typically FFT requires full grid. We pass the whole array.
        # But outside contact, shear is 0 (or noise).
        # We need to be careful if boundary affects FFT. 
        # For now, we follow original logic: decompose full image.
        
        U_divfree, V_divfree, _, _ = helmholtz_decompose(U_full, V_full)
        
        us = U_divfree[yy, xx]
        vs = V_divfree[yy, xx]
    else:
        us = shear_arr[yy, xx, 0]
        vs = shear_arr[yy, xx, 1]

    xs = X[yy, xx]
    ys = Y[yy, xx]
    zs = Z[yy, xx]
    
    # 4. Construct Linear System Ax = b
    # vector v = ω x r + t = | i  j  k | + |tx|
    #                        |wx wy wz |   |ty|
    #                        | x  y  z |   |tz| (tz usually ignored in 2D shear/flow?)
    #
    # v_x = wy*z - wz*y + tx
    # v_y = wz*x - wx*z + ty
    # v_z = wx*y - wy*x + tz (we don't measure Normal flow usually, just planar shear)
    
    # Unknowns: [wx, wy, wz] (and optionally tx, ty)
    
    n_pts = len(xs)
    # We have 2 equations per point (u, v)
    
    # Rows for X-equation (u): [0, z, -y] * [wx, wy, wz] = u - tx
    # Rows for Y-equation (v): [-z, 0, x] * [wx, wy, wz] = v - ty
    
    if include_translation:
        # Solve for [wx, wy, wz, tx, ty]
        # A is (2*N, 5)
        # u row: [0,  z, -y, 1, 0]
        # v row: [-z, 0,  x, 0, 1]
        
        A = np.zeros((2 * n_pts, 5), dtype=np.float32)
        b = np.zeros((2 * n_pts,), dtype=np.float32)
        
        # Fill A (interleave x/y rows)
        # row 2*k:   [0, z, -y, 1, 0]
        # row 2*k+1: [-z, 0, x, 0, 1]
        
        # Vectorized fill
        
        # wx cols
        A[0::2, 0] = 0.0
        A[1::2, 0] = -zs
        
        # wy cols
        A[0::2, 1] = zs
        A[1::2, 1] = 0.0
        
        # wz cols
        A[0::2, 2] = -ys
        A[1::2, 2] = xs
        
        # tx cols
        A[0::2, 3] = 1.0
        A[1::2, 3] = 0.0
        
        # ty cols
        A[0::2, 4] = 0.0
        A[1::2, 4] = 1.0
        
        b[0::2] = us
        b[1::2] = vs
        
    else:
        # Solve for [wx, wy, wz] only
        A = np.zeros((2 * n_pts, 3), dtype=np.float32)
        b = np.zeros((2 * n_pts,), dtype=np.float32)
        
        A[0::2, 0] = 0.0
        A[1::2, 0] = -zs
        
        A[0::2, 1] = zs
        A[1::2, 1] = 0.0
        
        A[0::2, 2] = -ys
        A[1::2, 2] = xs
        
        b[0::2] = us
        b[1::2] = vs

    # 5. Solve Least Squares
    # params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    # lstsq is slow for large matrices. Maybe use Cholesky on A.T @ A?
    # For N=1000 points, 2000 rows. It is fine.
    
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)

    omega = params[:3]
    omega_deg = np.degrees(omega)
    
    result = {
        'omega_rad': omega,
        'omega_deg': omega_deg,
        'rank': rank,
        's': s,
        'num_points': n_pts,
    }
    
    if len(residuals) > 0:
        result['residuals'] = residuals[0] # sum of squared residuals
    else:
        result['residuals'] = 0.0
        
    if include_translation:
        result['translation_m'] = params[3:5]
    else:
        result['translation_m'] = None

    return result
