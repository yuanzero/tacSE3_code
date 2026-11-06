import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from tacse3.core.estimator import pixel_xy_grid
from tacse3.core.robot_control import compute_action_from_pose, request_rotation
from tacse3.core.robot_worker import RobotWorker
from tacse3.offline_replay import metric_fields, replay


class OfflineTests(unittest.TestCase):
    def test_numerical_imports_do_not_load_sdk(self):
        self.assertNotIn('dmrobotics', sys.modules)
        self.assertNotIn('rtde_control', sys.modules)
        self.assertNotIn('lerobot', sys.modules)

    def test_unit_equivalence_and_known_rigid_motion(self):
        x,y=pixel_xy_grid((20,30),10.)
        z=.002+.0008*np.sin(x*600)*np.cos(y*550)
        w=np.array([.008,-.012,.02]);t=np.array([.00001,-.00002])
        s=np.stack((w[1]*z-w[2]*y+t[0],w[2]*x-w[0]*z+t[1]),axis=-1)
        for du,ds in [('m',1),('mm',1000)]:
            for su,ss in [('m',1),('mm',1000),('pixel',10000),('dmv1-code',5000)]:
                r=replay(z*ds,s*ss,depth_unit=du,shear_unit=su,ppmm=10.)
                np.testing.assert_allclose(r['rotation_vector_rad'],w,atol=1e-6)
                np.testing.assert_allclose(r['translation_xy_m'],t,atol=1e-8)
                self.assertFalse(r['physical_accuracy_validated'])

    def test_empty_and_degenerate_contact(self):
        z=np.zeros((8,8));s=np.zeros((8,8,2))
        kw=dict(depth_unit='m',shear_unit='m',ppmm=10.)
        self.assertEqual(replay(z,s,**kw)['status'],'insufficient_contact')
        self.assertIn('rank_deficient',replay(z+.002,s,**kw)['flags'])
        with self.assertRaises(ValueError):metric_fields(z,s,depth_unit='mm',shear_unit='pixel',ppmm=0)

    def test_limit_after_gain_and_matrix_path(self):
        for req in [dict(send_delta=True,omega_rad=[.03,0,0],delta_scale=100),
                    dict(R=Rotation.from_rotvec([.8,0,0]).as_matrix())]:
            req['max_step_deg']=.25
            self.assertAlmostEqual(np.degrees(Rotation.from_matrix(request_rotation(req)).magnitude()),.25)
        with self.assertRaises(ValueError):request_rotation(dict(send_delta=True,omega_rad=[1,0,0],delta_scale=float('inf')))

    def test_tool_tip_is_fixed_in_both_frames(self):
        p=np.array([.3,-.1,.4]);r=Rotation.from_euler('xyz',[.2,.3,.4]);offset=np.array([0,0,.25])
        change=Rotation.from_euler('x',.03)
        for frame in ['global','local']:
            a=compute_action_from_pose(p,r.as_rotvec(),change.as_matrix(),frame=frame,tool_offset_z=.25)
            newp=np.array([a['abs_x'],a['abs_y'],a['abs_z']])
            newr=Rotation.from_rotvec([a['abs_roll'],a['abs_pitch'],a['abs_yaw']])
            np.testing.assert_allclose(newp+newr.apply(offset),p+r.apply(offset),atol=1e-10)
            expected=change*r if frame=='global' else r*change
            np.testing.assert_allclose(newr.as_matrix(),expected.as_matrix(),atol=1e-10)

    def test_worker_with_memory_only_robot(self):
        class FakeRobot:
            is_connected=True
            def __init__(self):self.r_inter=self;self.actions=[];self.sent=threading.Event()
            def getActualTCPPose(self):return [0,0,0,0,0,0]
            def send_action(self,a):self.actions.append(a);self.sent.set()
        robot=FakeRobot();worker=RobotWorker(robot);worker.start()
        try:
            worker.send(dict(send_delta=True,omega_rad=[.1,0,0],delta_scale=20,max_step_deg=.2))
            self.assertTrue(robot.sent.wait(2))
            self.assertAlmostEqual(np.degrees(robot.actions[0]['abs_roll']),.2)
        finally:worker.stop();worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_observer_success_and_blocked_sdk(self):
        root=str(Path(__file__).resolve().parents[1])
        for mode in ['ok','blocked','stalled','invalid']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                p=Path(temp)
                (p/'dmrobotics.py').write_text('''import time, numpy as np
class Sensor:
 def __init__(self,serial):self.count=0
 def getStatus(self):return 0
 def getRawImage(self):return np.zeros((4,4))
 def getDepth(self):
  self.count+=1
  MODE
  return np.zeros((4,4))
 def getShear(self):return np.zeros((4,4,2))
 def disconnect(self):pass
'''.replace('MODE',{'ok':'pass','blocked':'time.sleep(60)','stalled':'time.sleep(60) if self.count>2 else None','invalid':"raise ValueError('synthetic invalid frame')"}[mode]))
                env=dict(os.environ,PYTHONPATH=temp+os.pathsep+root,OPENBLAS_NUM_THREADS='1')
                result=subprocess.run([sys.executable,'-B','-m','tacse3.observe','--sensor-serial','FAKE1',
                    '--duration','2' if mode=='stalled' else '.3','--startup-timeout','4','--stall-timeout','.4','--output',str(p/'session')],cwd=temp,env=env,
                    capture_output=True,text=True,timeout=15)
                report=json.loads((p/'session/summary.json').read_text())
                self.assertEqual(result.returncode,0 if mode=='ok' else 1,result.stdout+result.stderr)
                self.assertEqual(report['status'],'completed' if mode=='ok' else 'failed')
                if mode=='blocked':self.assertIn('startup timed out',report['error'])
                if mode=='stalled':self.assertIn('stream stalled',report['error'])
                self.assertTrue((p/'session/report.html').exists())


if __name__=='__main__':unittest.main()
