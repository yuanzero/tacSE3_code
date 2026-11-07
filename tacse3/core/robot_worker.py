import threading
import time
import logging
import numpy as np
from .math_utils import rotvec_to_matrix, euler_angles_to_rotation_matrix
from .robot_control import compute_action_from_pose, request_rotation

log = logging.getLogger("daimon_robot_worker")

class RobotWorker(threading.Thread):
    """Background worker that manages robot connection, fetches current TCP
    pose, computes actions and sends them. Receives estimation requests
    from the main loop (overwrite-latest semantics).
    """
    def __init__(self, robot):
        super().__init__(daemon=True)
        self.robot = robot
        # 不可命名为 _stop：会与 threading.Thread 内部 API 冲突，导致 join() 报错。
        self._stop_evt = threading.Event()
        # remember initial connected state to avoid double connect/disconnect
        self._initially_connected = bool(getattr(robot, 'is_connected', False))
        self.connected = self._initially_connected
        # latest request to process (overwrite semantics)
        self._latest_req = None
        self._evt = threading.Event()

    def send(self, req: dict):
        """Queue the latest estimation request (overwrite previous) and notify worker.

        Expected req keys:
          - 'R': rotation matrix (3x3) OR 'omega_rad': 3-vector; prefer R when present
          - 'send_delta': bool
          - 'delta_scale': float
          - 'multiply': 'left'|'right'
          - 'frame': 'global' (base) | 'local' (tool/TCP): passed to compute_action_from_pose
        """
        self._latest_req = req
        self._evt.set()

    def stop(self):
        self._stop_evt.set()
        self._evt.set()

    def run(self):
        try:
            if not self._initially_connected:
                try:
                    self.robot.connect()
                    self.connected = True
                    log.info('RobotWorker connected')
                except Exception as e:
                    log.warning('RobotWorker initial connect failed: %s', e)
        except Exception as e:
            log.warning('RobotWorker setup failed: %s', e)
        while not self._stop_evt.is_set():
            try:
                got = self._evt.wait(timeout=0.2)
            except Exception:
                got = False
            if not got:
                continue
            self._evt.clear()
            if self._stop_evt.is_set():
                break
            req = self._latest_req
            if req is None:
                continue
            # ensure connected
            if not self.connected:
                try:
                    self.robot.connect()
                    self.connected = True
                    log.info('RobotWorker connected')
                except Exception as e:
                    log.warning('RobotWorker connect failed: %s', e)
                    # skip this request; wait for next
                    continue
            # fetch current pose and compute action here to avoid blocking main thread
            try:
                cur = self.robot.r_inter.getActualTCPPose()
            except Exception as e:
                log.warning('RobotWorker getActualTCPPose failed: %s', e)
                continue
            try:
                R_use = request_rotation(req)
                # Now apply this rotation (whether delta or absolute) using the correct frame/offset logic
                action = compute_action_from_pose(
                    cur[:3], 
                    cur[3:6], 
                    R_use, 
                    multiply=req.get('multiply', 'left'), 
                    frame=req.get('frame', 'global'),
                    tool_offset_z=float(req.get('tool_offset_z', 0.0))
                )
                
                log.debug('RobotWorker computed action: %s', action)
                try:
                    self.robot.send_action(action)
                    log.debug('RobotWorker send_action returned')
                except Exception as e:
                    log.warning('RobotWorker send_action failed: %s', e)
            except Exception as e:
                log.warning('RobotWorker failed to compute/send action: %s', e)
        # cleanup: only disconnect if we connected here
        try:
            if not self._initially_connected and self.connected:
                self.robot.disconnect()
        except Exception:
            pass
