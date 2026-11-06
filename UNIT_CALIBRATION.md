# Unit conventions

[Project overview](README.md) · [Architecture](docs/architecture.md)

TacSE3 converts sensor inputs to meters before numerical estimation. Display
units and dimensionless gains are separate from that conversion.

| Quantity | Convention | Conversion to estimator input |
| --- | --- | --- |
| Image scale | `mm_per_pixel`, nominal default 0.0634 | `ppmm = 1 / mm_per_pixel` |
| Estimator `ppmm` | Actual pixels per millimeter | Pixel coordinates × `0.001 / ppmm` |
| Depth | Nominal millimeters for the DMV1 profile | × 0.001 |
| DMV1 shear | SDK-scaled displacement | × 0.0002 |
| Rotation vector | Radians | Degrees are used only for presentation |
| Translation | Meters | Convert explicitly when presenting other units |

## SDK profile

`core/units.py` defines the `dmv1-code` profile. It corresponds to DMV1 SDK SHA256
`d8ad98da0f37b129a1e5fcee20d8a6730e4523c5353b55dc45adc11346acda17`.
Its numerical convention is pixel flow × internal mm/pixel × 100 × 0.05;
therefore nominal displacement in meters is the returned shear value × 0.0002.
The SDK internal pitch and the output-image pitch are separate parameters.
Shear displacement is not force or pressure.

Use this profile with matching SDK data. Other adapters can supply explicit
meter, millimeter or pixel units to the replay API. Nominal defaults describe a
software convention; applications requiring physical measurement should supply
sensor-specific calibration.

## APIs

- `metric_fields` accepts `depth_unit`, `shear_unit` and actual `ppmm` explicitly.
- `build_contact_from_daimon` defaults to nominal mm depth and `dmv1-code` shear.
  For metric arrays pass `depth_in_mm=False, shear_unit='m'`.
- `estimate_rotation_from_shear` expects depth and shear in meters.
- `shear_scale` is a separate, dimensionless gain applied after unit conversion.

```bash
python -m tacse3.offline_replay sample.npz \
  --depth-unit mm --shear-unit dmv1-code --mm-per-pixel 0.0634 --output replay.json
```

The replay output records its unit assumptions, profile hash, fit diagnostics
and physical-calibration status. Selecting a profile identifies the convention
of the input data; it does not infer calibration from a saved image.
