import io
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import torch
from tactile_recorder.features import ImageFeatures, hog, points_from_depth, pcd_bytes
from tactile_recorder.recording import Recording
from tactile_recorder.server import Engine, handler_for
from utilities.reconstruction import Reconstruction3D


def args(output, **kwargs):
    values = dict(demo=True, no_depth=False, model=Path('models/nnmini.pt'), border=.15,
                  fps=30, output=Path(output), device='/dev/video0')
    values.update(kwargs)
    return Namespace(**values)


def packet(engine, image=None):
    engine.source_frame += 1
    raw, depth = engine.demo_frame()
    if image is not None:
        raw = image
    return engine.process(raw, depth, time.time_ns(), time.time_ns(), time.monotonic_ns())


class FeaturesTest(unittest.TestCase):
    def test_reference_freezes_and_input_is_not_modified(self):
        processor = ImageFeatures()
        base = np.full((32, 32, 3), 100, np.uint8)
        before = base.copy()
        processor.update(base)
        numeric, _, _ = processor.update(base)
        np.testing.assert_array_equal(numeric['image_gradient_magnitude'], 0)
        self.assertTrue(np.isfinite(numeric['image_hog_descriptor']).all())
        np.testing.assert_array_equal(base, before)
        contact = base.copy()
        contact[:, 16:] = 200
        numeric, arrows, _ = processor.update(contact)
        self.assertGreater(numeric['image_gradient_magnitude'].max(), .1)
        np.testing.assert_allclose(numeric['image_reference_gray'], 100/255)
        self.assertTrue(np.any(arrows != contact))

    def test_hog_direction_and_normalization(self):
        gray = np.tile(np.arange(32, dtype=float), (32, 1))
        result = hog(gray)
        self.assertEqual(result['image_hog_cells'].shape, (4, 4, 9))
        self.assertEqual(np.argmax(result['image_hog_histograms'].sum(axis=(0, 1))), 0)
        np.testing.assert_allclose(np.linalg.norm(result['image_hog_blocks'].reshape(9, -1), axis=1), 1)

    def test_pcd_coordinates_and_binary_roundtrip(self):
        depth = np.array([[1, 2, 3], [-1, -2, -3]], np.float32)
        points = points_from_depth(depth)
        np.testing.assert_array_equal(points[1, 2], [2, 1, -3])
        header, data = pcd_bytes(points).split(b'DATA binary\n')
        self.assertIn(b'POINTS 6', header)
        np.testing.assert_array_equal(np.frombuffer(data, '<f4').reshape(2, 3, 3), points)


class ReconstructionTest(unittest.TestCase):
    def test_fractional_pixel_coordinates_and_finite_normals(self):
        class Net(torch.nn.Module):
            def forward(self, inputs):
                self.inputs = inputs.clone()
                return torch.ones((len(inputs), 2)) * .2
        model = Reconstruction3D(8, 6)
        model.net = Net()
        depth, _, gx, gy = model.get_depthmap(np.full((6, 8, 3), 100, np.uint8))
        self.assertGreater(model.net.inputs[:, 3:].max(), .5)
        self.assertLess(model.net.inputs[:, 3:].max(), 1)
        self.assertTrue(all(np.isfinite(a).all() for a in (depth, gx, gy)))

    def test_invalid_normal_does_not_create_thousandfold_slope(self):
        class Net(torch.nn.Module):
            def forward(self, inputs):
                result = torch.full((len(inputs), 2), .2)
                result[0] = torch.tensor([1.2, .4])
                return result
        model = Reconstruction3D(8, 6)
        model.net = Net()
        _, _, gx, gy = model.get_depthmap(np.full((6, 8, 3), 100, np.uint8))
        self.assertLess(np.abs(gx).max(), 2)
        self.assertLess(np.abs(gy).max(), 2)
        self.assertAlmostEqual(model.invalid_normal_fraction, 1/48)

    def test_all_invalid_normals_reject_frame_without_calibrating(self):
        class Net(torch.nn.Module):
            def forward(self, inputs):
                return torch.full((len(inputs), 2), 2.)
        model = Reconstruction3D(8, 6)
        model.net = Net()
        with self.assertRaisesRegex(ValueError, 'no usable surface normals'):
            model.get_depthmap(np.full((6, 8, 3), 100, np.uint8))
        self.assertEqual(model.depth_map_zero_counter, 0)

    def test_marker_mask_with_no_valid_pixels_rejects_depth(self):
        model = Reconstruction3D(8, 6)
        with self.assertRaisesRegex(ValueError, 'No valid image pixels'):
            model.get_depthmap(np.zeros((6, 8, 3), np.uint8), markers_threshold=(0,70))

    def test_failed_depth_keeps_image_recording_available(self):
        class FailedModel:
            depth_map_zero_counter = 0
            def get_depthmap(self, *args, **kwargs):
                raise ValueError('Invalid surface normals')
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(args(temp))
            engine.args.demo = False
            engine.model = FailedModel()
            packet(engine)
            frame = packet(engine)
            self.assertTrue(frame['ready'])
            self.assertNotIn('depth', frame['numeric'])
            self.assertIn('image_hog_descriptor', frame['numeric'])
            self.assertEqual(engine.depth_error, 'Invalid surface normals')

    def test_calibration_subtracts_average_not_sum(self):
        class Net(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros((len(inputs), 2))
        model = Reconstruction3D(8, 6)
        model.net = Net()
        surface = np.arange(48).reshape(6, 8).astype(float)
        with patch('utilities.reconstruction.poisson_dct_neumann', side_effect=lambda **kw: surface.copy()):
            for _ in range(52):
                depth, _, _, _ = model.get_depthmap(np.zeros((6, 8, 3), np.uint8))
                np.testing.assert_allclose(depth, 0, atol=1e-6)

    def test_bundled_model_returns_valid_depth(self):
        torch.set_num_threads(2)
        model = Reconstruction3D(32, 24)
        self.assertIsNotNone(model.load_nn('models/nnmini.pt'))
        frame = np.full((24, 32, 3), (130, 145, 155), np.uint8)
        for _ in range(51):
            result = model.get_depthmap(frame, markers_threshold=(0, 70))
        self.assertTrue(np.isfinite(result[0]).all())
        np.testing.assert_allclose(result[0], 0, atol=1e-5)


class RecordingTest(unittest.TestCase):
    def test_lossless_session_and_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(args(temp))
            packet(engine)
            frame = packet(engine)
            recording = Recording(temp, 'test contact', engine.metadata())
            recording.save(frame)
            recording.close()
            path = recording.path
            np.testing.assert_array_equal(cv2.imread(str(path/'raw/000000.png')), frame['images']['raw'])
            with np.load(path/'numeric/000000.npz', allow_pickle=False) as numeric:
                np.testing.assert_array_equal(numeric['depth'], np.load(path/'depth/000000.npy'))
                np.testing.assert_array_equal(numeric['points'][..., 2], numeric['depth'])
                self.assertEqual(int(numeric['source_frame']), frame['source_frame'])
                self.assertEqual(int(numeric['capture_end_ns']), frame['capture_end_ns'])
            metadata = json.loads((path/'session.json').read_text())
            self.assertEqual(metadata['saved_frames'], 1)
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(len((path/'frames.csv').read_text().splitlines()), 2)

    def test_no_depth_mode_still_records_features(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(args(temp, no_depth=True))
            packet(engine)
            frame = packet(engine)
            recording = Recording(temp, '', engine.metadata())
            recording.save(frame)
            recording.close()
            self.assertFalse(list((recording.path/'depth').iterdir()))
            with np.load(recording.path/'numeric/000000.npz') as numeric:
                self.assertIn('image_hog_descriptor', numeric.files)
                self.assertNotIn('depth', numeric.files)

    def test_failed_write_is_not_committed(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(args(temp))
            recording = Recording(temp, '', engine.metadata())
            with patch('tactile_recorder.recording.cv2.imwrite', return_value=False):
                with self.assertRaises(OSError):
                    recording.save(packet(engine))
            recording.close('disk write failed')
            self.assertEqual(recording.count, 0)
            self.assertEqual(len((recording.path/'frames.csv').read_text().splitlines()), 1)
            self.assertEqual(json.loads((recording.path/'session.json').read_text())['status'], 'error')

    def test_calibration_cannot_change_mid_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(args(temp))
            engine.apply_source(engine.source_request)
            engine.source_request = None
            initial_epoch = engine.epoch
            with self.assertRaises(ValueError):
                engine.command('start')
            packet(engine)
            engine.packet = packet(engine)
            engine.command('start')
            with self.assertRaises(ValueError):
                engine.command('start')
            with self.assertRaises(ValueError):
                engine.command('calibrate')
            engine.command('stop')
            engine.command('calibrate')
            self.assertIsNone(engine.packet)
            self.assertEqual(engine.epoch, initial_epoch + 1)
            self.assertFalse(engine.status()['ready'])


class HTTPTest(unittest.TestCase):
    def test_recording_controls_downloads_and_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(args(temp))
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(engine))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            engine.thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def get(path):
                with urlopen(base+path, timeout=10) as response:
                    return response.read()
            def post(action, origin=None, payload=None):
                headers = {'Content-Type':'application/json'}
                if origin:
                    headers['Origin'] = origin
                data = json.dumps(payload or {'label':'integration'}).encode()
                with urlopen(Request(base+'/api/'+action, data=data, headers=headers), timeout=10) as response:
                    return json.load(response)
            try:
                deadline = time.monotonic()+10
                while not engine.status()['ready'] and time.monotonic() < deadline:
                    time.sleep(.03)
                self.assertTrue(engine.status()['ready'], engine.error)
                self.assertIn(b'<html lang="en">', get('/'))
                self.assertIsInstance(json.loads(get('/api/cameras'))['cameras'], list)
                self.assertTrue(post('start')['ok'])
                time.sleep(.3)
                self.assertTrue(json.loads(get('/api/state'))['recording'])
                self.assertTrue(post('stop')['ok'])
                path = Path(engine.last_recording)
                self.assertGreater(json.loads((path/'session.json').read_text())['saved_frames'], 0)
                self.assertEqual(np.load(io.BytesIO(get('/api/depth.npy'))).shape, (240, 320))
                self.assertIn(b'DATA binary\n', get('/api/cloud.pcd'))
                with np.load(io.BytesIO(get('/api/frame.npz'))) as numeric:
                    self.assertIn('raw_bgr', numeric.files)
                with self.assertRaises(HTTPError) as error:
                    post('start', 'http://unrelated.example')
                self.assertEqual(error.exception.code, 403)
                self.assertTrue(post('calibrate')['ok'])
                self.assertTrue(post('source', payload={'device':'__demo__', 'depth_enabled':True})['ok'])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                engine.close()


if __name__ == '__main__':
    unittest.main()
