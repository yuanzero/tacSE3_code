"""Numerical SDK probe. No DMV1 constructor, camera, thread or actuator is started."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
import threading

def forbidden(*args, **kwargs):
    raise RuntimeError('Hardware/thread creation forbidden in unit probe')

cv2.VideoCapture = forbidden
threading.Thread.start = forbidden
import dmrobotics.dmSDK as sdk

EXPECTED = 'd8ad98da0f37b129a1e5fcee20d8a6730e4523c5353b55dc45adc11346acda17'
digest = hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest()
assert digest == EXPECTED, 'SDK changed: re-audit conversion before using this profile'
rows = []
for mmpp in [.044692, .0634, .123]:
    p = SimpleNamespace()
    p._DMV1__stop = False
    p.pyarmor__19 = True
    p.pyarmor__21 = False
    p.get_sensor_state = lambda: 0
    p._DMV1__mmpp = mmpp
    p._DMV1__fps_calculation_active = False
    p._DMV1__boarder_size = 2
    class Camera:
        def read(self):
            p._DMV1__stop = True
            return True, np.zeros((48, 64, 3), np.uint8)
        def isOpened(self):
            return True
    class Decomposer:
        def decompose(self, field):
            self.h = field
            self.r = np.zeros_like(field)
    p._DMV1__cap = Camera()
    p._DMV1__transform = lambda x: x
    p._DMV1__t = SimpleNamespace(t=lambda x: np.ones((*x.shape, 2), np.float32))
    p._DMV1__shear_obj = Decomposer()
    sdk.DMV1._DMV1__update_image(p)
    shear = sdk.DMV1.getShear(p)
    np.testing.assert_allclose(p._DMV1__f, 100 * mmpp, rtol=1e-6)
    np.testing.assert_allclose(shear, 5 * mmpp, rtol=1e-6)
    rows.append(dict(injected_mm_per_pixel=mmpp, injected_flow_pixels=1,
                     internal_field=float(p._DMV1__f.mean()), returned_shear=float(shear.mean())))
print(json.dumps(dict(sdk_sha256=digest, cases=rows,
    interpretation='getShear = 5 * mmpp * decomposed optical-flow pixels; nominal mm = raw / 5',
    physical_accuracy_validated=False,
    limitations='Synthetic tracker and decomposer; does not measure sensor pixel pitch, axis alignment, depth accuracy or tracking accuracy.',
    actuator_commands_sent=0), indent=2))
