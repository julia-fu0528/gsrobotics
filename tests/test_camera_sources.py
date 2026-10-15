import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from tactile_recorder.cameras import list_cameras, DEMO_DEVICE
from tactile_recorder.server import Engine
from test_tactile_recorder import args

CAMERAS = [dict(device='/dev/v4l/by-id/sensor-a',path='/dev/video4',name='GelSight Mini A',label='GelSight Mini A (/dev/video4)',gelsight=True),
           dict(device='/dev/v4l/by-id/sensor-b',path='/dev/video6',name='USB camera B',label='USB camera B (/dev/video6)',gelsight=False)]


def wait_for(predicate):
    deadline=time.monotonic()+6
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.02)
    raise AssertionError('Timed out waiting for camera state')


class FakeCapture:
    def __init__(self,device):
        self.device=device;self.released=False;self.failed=False
        self.color=80 if str(device).endswith('sensor-a') else 160
    def isOpened(self):return True
    def set(self,*args):return True
    def read(self):
        if self.failed:return False,None
        return True,np.full((48,64,3),self.color,np.uint8)
    def release(self):self.released=True


class CameraDiscoveryTest(unittest.TestCase):
    def test_metadata_nodes_filtered_stable_path_preferred_and_hotplug_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sys=root/'sys';dev=root/'dev'
            aliases=dev/'v4l/by-id';aliases.mkdir(parents=True)
            for node,name,index in [('video4','GelSight Mini serial',0),('video5','GelSight Mini serial',1),('video6','Webcam',0)]:
                entry=sys/node;entry.mkdir(parents=True)
                (entry/'name').write_text(name);(entry/'index').write_text(str(index))
                (dev/node).touch()
            (aliases/'usb-gelsight-video-index0').symlink_to(dev/'video4')
            cameras=list_cameras(sys,dev)
            self.assertEqual(len(cameras),2)
            self.assertTrue(cameras[0]['gelsight'])
            self.assertEqual(cameras[0]['device'],str(aliases/'usb-gelsight-video-index0'))
            self.assertFalse(cameras[1]['gelsight'])
            (sys/'video4/name').unlink()
            self.assertEqual(len(list_cameras(sys,dev)),1)


class CameraSwitchTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.discovery=patch('tactile_recorder.server.list_cameras',return_value=CAMERAS)
        self.discovery.start()
        self.captures=[]
        def create(device):
            cap=FakeCapture(device);self.captures.append(cap);return cap
        self.capture_patch=patch('tactile_recorder.server.cv2.VideoCapture',side_effect=create)
        self.capture_patch.start()
        self.engine=Engine(args(self.temp.name,demo=False,device=None,no_depth=True,fps=20))
        self.engine.thread.start()
    def tearDown(self):
        self.engine.close();self.capture_patch.stop();self.discovery.stop();self.temp.cleanup()

    def connect(self,index=0):
        self.engine.select_source(CAMERAS[index]['device'],False)
        wait_for(lambda:self.engine.status()['ready'])

    def test_idle_start_switch_and_reference_reset(self):
        self.assertFalse(self.engine.status()['connected'])
        self.assertEqual(self.captures,[])
        self.connect(0)
        first=self.engine.packet
        epoch=first['epoch']
        self.engine.select_source(CAMERAS[1]['device'],False)
        wait_for(lambda:self.engine.status()['ready'] and self.engine.args.device==CAMERAS[1]['device'])
        self.assertTrue(self.captures[0].released)
        self.assertGreater(self.engine.packet['epoch'],epoch)
        np.testing.assert_array_equal(self.engine.packet['images']['raw'],160)
        np.testing.assert_allclose(self.engine.packet['numeric']['image_reference_bgr'],160/255,atol=1e-7)
        np.testing.assert_allclose(self.engine.packet['numeric']['image_delta_bgr'],0,atol=1e-7)

    def test_unplug_keeps_worker_alive_and_finalizes_recording(self):
        self.connect()
        self.engine.command('start')
        self.captures[0].failed=True
        wait_for(lambda:self.engine.error is not None)
        self.assertTrue(self.engine.thread.is_alive())
        self.assertFalse(self.engine.status()['recording'])
        self.assertIsNone(self.engine.packet)
        metadata=json.loads((Path(self.engine.last_recording)/'session.json').read_text())
        self.assertEqual(metadata['status'],'error')
        self.connect(1)
        self.assertIsNone(self.engine.error)
        self.assertTrue(self.engine.status()['ready'])

    def test_reject_switch_during_recording_and_arbitrary_sources(self):
        self.connect()
        self.engine.command('start')
        with self.assertRaisesRegex(ValueError,'Stop recording'):
            self.engine.select_source(CAMERAS[1]['device'],False)
        self.engine.command('stop')
        with self.assertRaisesRegex(ValueError,'no longer available'):
            self.engine.select_source('https://example.com/movie',False)
        with self.assertRaises(ValueError):self.engine.select_source(None,False)

    def test_disconnect_then_connect_demo(self):
        self.connect()
        self.engine.select_source('',False)
        wait_for(lambda:not self.engine.switching)
        self.assertFalse(self.engine.status()['connected'])
        self.assertIsNone(self.engine.packet)
        self.assertTrue(self.captures[0].released)
        self.engine.select_source(DEMO_DEVICE,True)
        wait_for(lambda:self.engine.status()['ready'])
        self.assertTrue(self.engine.args.demo)
        self.assertTrue(self.engine.status()['depth_available'])

    def test_failed_open_can_be_retried(self):
        with patch.object(FakeCapture,'isOpened',return_value=False):
            self.engine.select_source(CAMERAS[0]['device'],False)
            wait_for(lambda:self.engine.error is not None)
        self.assertTrue(self.engine.thread.is_alive())
        self.assertTrue(self.captures[0].released)
        self.connect()
        self.assertIsNone(self.engine.error)

    def test_in_flight_old_frame_is_discarded_on_switch(self):
        self.connect()
        old_capture=self.captures[0]
        entered=threading.Event();resume=threading.Event()
        def blocked_read():
            entered.set();resume.wait(3)
            return True,np.full((48,64,3),80,np.uint8)
        old_capture.read=blocked_read
        self.assertTrue(entered.wait(3))
        try:
            self.engine.select_source(CAMERAS[1]['device'],False)
            self.assertIsNone(self.engine.packet)
        finally:resume.set()
        wait_for(lambda:self.engine.status()['ready'] and self.engine.args.device==CAMERAS[1]['device'])
        np.testing.assert_array_equal(self.engine.packet['images']['raw'],160)
        self.assertTrue(old_capture.released)


if __name__=='__main__':unittest.main()
