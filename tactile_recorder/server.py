"""Run python -m tactile_recorder.server, then select a camera in the browser."""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import cv2
import numpy as np
from .features import ImageFeatures, points_from_depth, pcd_bytes
from .recording import Recording
from .cameras import DEMO_DEVICE, list_cameras, camera_for_device

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).parent / 'ui'


class Engine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.features = ImageFeatures()
        self.epoch = 0
        self.packet = None
        self.preview = None
        self.recording = None
        self.last_recording = None
        self.last_count = 0
        self.error = None
        self.depth_error = None
        self.model = None
        self.source_frame = 0
        self.fps = 0.
        self.cap = None
        self.camera_info = camera_for_device(args.device, list_cameras())
        initial_depth = not args.no_depth and (args.demo or bool(self.camera_info and self.camera_info['gelsight']))
        self.source_revision = 0
        self.switching = bool(args.demo or args.device)
        self.source_request = (args.device, args.demo, initial_depth, self.camera_info, 0)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def status(self):
        with self.lock:
            age = None if self.packet is None else (time.monotonic_ns()-self.packet['monotonic_ns'])/1e9
            ready = self.packet is not None and self.packet['ready'] and age < 2 and not self.error and not self.switching
            return dict(mode='Demo · synthetic data' if self.args.demo else (self.camera_info['label'] if self.camera_info else self.args.device or 'No camera selected'),
                        source_device=DEMO_DEVICE if self.args.demo else self.args.device,
                        switching=self.switching, depth_enabled=not self.args.no_depth,
                        connected=age is not None and age < 2 and not self.error,
                        ready=bool(ready), frame_age_seconds=age, source_frame=self.source_frame,
                        fps=round(self.fps, 1), recording=self.recording is not None,
                        saved_frames=self.recording.count if self.recording else self.last_count,
                        session=str(self.recording.path) if self.recording else self.last_recording,
                        error=self.error, depth_error=self.depth_error, reference_epoch=self.epoch,
                        calibration_frames=min(self.model.depth_map_zero_counter, 50) if self.model else min(len(self.features.references), 2),
                        calibration_total=50 if self.model else 2,
                        depth_available=bool(ready and 'depth' in self.packet['numeric']),
                        depth_units='model units (uncalibrated)', preview=self.preview)

    def metadata(self):
        return dict(source='synthetic_demo' if self.args.demo else str(self.args.device),
                    camera_name=self.camera_info['name'] if self.camera_info else None,
                    depth_enabled=not self.args.no_depth,
                    model=str(self.args.model), depth_method='synthetic' if self.args.demo else 'RGB2NormNet + Poisson',
                    processing_size=[320, 240], border_fraction=self.args.border,
                    raw='Unmodified OpenCV BGR camera frame, stored as lossless PNG; not Bayer data.',
                    image='Crop and resize of raw using GS crop_and_resize; RGB input to depth network.',
                    image_features='Grayscale /255, subtract frozen mean of two frames, np.gradient; x=columns, y=rows.',
                    deformation_image='Signed BGR difference from the frozen mean of the first two sensing-image frames; PNG = clip(round(128 + 255 * image_delta_bgr), 0, 255). Neutral gray means unchanged. NPZ reference/delta arrays use float32 normalized intensity.',
                    hog='8px cells, 9 unsigned bins, interpolated voting, 2x2 L2-Hys blocks; on reference-subtracted sensing image.',
                    depth='Signed relative model depth after 50 untouched frames; not camera range or calibrated millimeters.',
                    points='Organized float32 HxWx3 XYZ: x=column, y=row, z=depth; PCD flattens row-major.',
                    surface_gradients='Raw model surface slopes before zero-depth subtraction; distinct from image gradients.',
                    reference_epoch=self.epoch, requested_hz=self.args.fps,
                    timing='All products share one acquired frame. CSV timestamps are host read start/end, not hardware timestamps. Actual rate may be lower than requested; no duplication.',
                    numeric='NPZ contains image reference, image gradients, HOG arrays, and when available depth, surface gradients, valid mask, points. Depth NPY and binary XYZ PCD duplicate those arrays.',
                    depth_preview='Symmetric per-frame range, at least +/-1 model unit; depth_display_limit records the range. Numeric depth is unclipped.',
                    normal_handling='Upstream mean-valid-Z fallback for invalid normals; invalid_normal_fraction records the repaired proportion. All-invalid frames omit depth.')

    def command(self, action, label=''):
        with self.lock:
            if action == 'start':
                if self.recording:
                    raise ValueError('A recording is already running')
                if not self.status()['ready']:
                    raise ValueError('Wait for a live frame and reference calibration')
                self.recording = Recording(self.args.output, label[:200], self.metadata())
            elif action == 'stop':
                self.finish()
            elif action == 'calibrate':
                if self.switching:
                    raise ValueError('Wait for the camera connection to finish')
                if self.recording:
                    raise ValueError('Stop recording before changing the reference')
                self.reset_reference()
            else:
                raise ValueError('Unknown action')

    def reset_reference(self):
        self.features = ImageFeatures()
        self.epoch += 1
        self.packet = self.preview = None
        if self.model:
            self.model.depth_map_zero.fill(0)
            self.model.depth_map_zero_counter = 0

    def select_source(self, device, depth_enabled=False):
        if not isinstance(device, str) or not isinstance(depth_enabled, bool):
            raise ValueError('Select a camera and a valid depth setting')
        demo = device == DEMO_DEVICE
        info = None
        if device and not demo:
            info = next((c for c in list_cameras() if c['device'] == device), None)
            if info is None:
                raise ValueError('Camera is no longer available. Refresh the camera list.')
        with self.lock:
            if self.recording:
                raise ValueError('Stop recording before changing the camera')
            self.source_revision += 1
            self.source_request = (None if demo or not device else device, demo, depth_enabled, info, self.source_revision)
            self.switching = True
            self.packet = self.preview = None
            self.error = self.depth_error = None
            self.fps = 0.

    def apply_source(self, request):
        """Only the acquisition thread owns camera open/read/release operations."""
        device, demo, depth_enabled, info, revision = request
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = model = None
        error = depth_error = None
        try:
            if device and not demo:
                cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
                if not cap.isOpened():
                    raise RuntimeError(f'Cannot open camera {device}. Close other camera viewers, refresh, and reconnect.')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if device and depth_enabled and not demo:
                try:
                    from utilities.reconstruction import Reconstruction3D
                    import torch
                    torch.set_num_threads(2)
                    model = Reconstruction3D(320, 240)
                    if model.load_nn(str(self.args.model)) is None:
                        raise RuntimeError('Depth model was not found')
                except Exception as exc:
                    model, depth_error = None, str(exc)
        except Exception as exc:
            error = str(exc)
            if cap is not None:
                cap.release()
                cap = None
        with self.lock:
            if revision != self.source_revision or self.stop_event.is_set():
                if cap is not None:
                    cap.release()
                return
            self.cap, self.model = cap, model
            self.args.device, self.args.demo = device, demo
            self.args.no_depth = not depth_enabled
            self.camera_info = info
            self.error, self.depth_error = error, depth_error
            self.reset_reference()
            self.source_frame = 0
            self.switching = False

    def finish(self, error=None):
        if self.recording:
            recording, self.recording = self.recording, None
            self.last_recording = str(recording.path)
            self.last_count = recording.count
            recording.close(error)

    def demo_frame(self):
        y, x = np.mgrid[:240, :320]
        # Untouched start, followed by a moving synthetic indentation.
        t = max(0, self.source_frame-12)/self.args.fps
        depth = -5*np.exp(-((x-160-55*np.sin(t))**2+(y-120)**2)/1000) * min(1, t)
        dy, dx = np.gradient(depth)
        bgr = np.stack((130+dx*210, 145+dy*210, 155-dx*160-dy*100), axis=-1)
        return np.clip(bgr, 0, 255).astype(np.uint8), depth.astype(np.float32)

    def process(self, raw, synthetic_depth, started, ended, monotonic):
        from utilities.image_processing import crop_and_resize
        image = raw.copy() if self.args.demo else crop_and_resize(raw, (320, 240), self.args.border)
        numeric, arrows, glyphs = self.features.update(image)
        difference = np.clip(np.rint(128+255*numeric['image_delta_bgr']), 0, 255).astype(np.uint8)
        images = dict(raw=raw, image=image, deformation=difference, gradient=arrows, hog=glyphs)
        ready = len(self.features.references) == 2
        depth = None
        if not self.args.no_depth:
            if self.args.demo:
                depth = synthetic_depth
                sy, sx = np.gradient(depth)
                mask = np.ones(depth.shape, dtype=bool)
            elif self.model:
                try:
                    depth, mask, sx, sy = self.model.get_depthmap(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers_threshold=(0, 70))
                    if not all(np.isfinite(a).all() for a in (depth, sx, sy)):
                        raise ValueError('Depth reconstruction returned non-finite values')
                    self.depth_error = None
                except Exception as exc:
                    self.depth_error = str(exc)
                    depth = None
                # A failed depth estimate must not disable image-only recording.
                ready = ready and (self.depth_error is not None or self.model.depth_map_zero_counter >= 50)
            if depth is not None and ready:
                numeric.update(depth=depth.astype(np.float32), surface_gradient_x=sx.astype(np.float32),
                               surface_gradient_y=sy.astype(np.float32), valid_mask=mask,
                               points=points_from_depth(depth))
                limit = max(1., float(np.max(np.abs(depth))))
                numeric['depth_display_limit'] = np.float32(limit)
                numeric['invalid_normal_fraction'] = np.float32(self.model.invalid_normal_fraction if self.model else 0)
                images['depth_preview'] = cv2.applyColorMap(np.clip((depth/limit+1)*127.5, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        numeric.update(source_frame=np.int64(self.source_frame), capture_start_ns=np.int64(started),
                       capture_end_ns=np.int64(ended), reference_epoch=np.int64(self.epoch))
        return dict(images=images, numeric=numeric, ready=ready, epoch=self.epoch,
                    source_frame=self.source_frame, capture_start_ns=started, capture_end_ns=ended, monotonic_ns=monotonic)

    def run(self):
        try:
            while not self.stop_event.is_set():
                tick = time.monotonic()
                with self.lock:
                    request, self.source_request = self.source_request, None
                if request is not None:
                    self.apply_source(request)
                with self.lock:
                    revision = self.source_revision
                    demo = self.args.demo
                    idle = self.switching or (not demo and self.cap is None)
                if idle:
                    self.stop_event.wait(.1)
                    continue
                try:
                    started = time.time_ns()
                    if demo:
                        raw, depth = self.demo_frame()
                    else:
                        ok, raw = self.cap.read()
                        if not ok:
                            raise RuntimeError('Camera disconnected or frame read failed. Refresh cameras and reconnect.')
                        depth = None
                    ended, monotonic = time.time_ns(), time.monotonic_ns()
                    with self.lock:
                        # A read may finish after the user requested a different source.
                        if revision != self.source_revision or self.switching:
                            continue
                        self.source_frame += 1
                        packet = self.process(raw, depth, started, ended, monotonic)
                        encoded = {}
                        for key, value in packet['images'].items():
                            ok, jpeg = cv2.imencode('.jpg', value, [cv2.IMWRITE_JPEG_QUALITY, 82])
                            if not ok:
                                raise RuntimeError('Preview encoding failed')
                            encoded[key] = base64.b64encode(jpeg).decode('ascii')
                        self.preview = dict(images=encoded, source_frame=self.source_frame, points=None)
                        if 'depth' in packet['numeric']:
                            self.preview['points'] = packet['numeric']['points'][::5, ::5].reshape(-1, 3).round(3).tolist()
                            self.preview['depth_limit'] = float(packet['numeric']['depth_display_limit'])
                            self.preview['depth_range'] = [float(packet['numeric']['depth'].min()), float(packet['numeric']['depth'].max())]
                            self.preview['invalid_normal_fraction'] = float(packet['numeric']['invalid_normal_fraction'])
                        self.packet = packet
                        if self.recording:
                            try:
                                self.recording.save(packet)
                            except Exception as exc:
                                self.error = f'Recording stopped: {exc}'
                                self.finish(self.error)
                except Exception as exc:
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    with self.lock:
                        if revision == self.source_revision:
                            self.error = str(exc)
                            self.packet = self.preview = None
                            self.fps = 0.
                            self.finish(self.error)
                self.stop_event.wait(max(0, 1/self.args.fps-(time.monotonic()-tick)))
                self.fps = 1/max(1e-6, time.monotonic()-tick)
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        with self.lock:
            self.finish()


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, data, kind='application/json', code=200, filename=None):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/state':
                self.send(json.dumps(engine.status()).encode())
            elif path == '/api/cameras':
                self.send(json.dumps({'cameras': list_cameras(), 'demo_device': DEMO_DEVICE}).encode())
            elif path in ('/api/frame.npz', '/api/depth.npy', '/api/cloud.pcd'):
                with engine.lock:
                    packet = engine.packet
                    ready = engine.status()['ready']
                if not ready or (path != '/api/frame.npz' and 'depth' not in packet['numeric']):
                    self.send(b'{"error":"Current data unavailable"}', code=409)
                    return
                stream = io.BytesIO()
                if path.endswith('.npz'):
                    np.savez_compressed(stream, raw_bgr=packet['images']['raw'], image_bgr=packet['images']['image'], **packet['numeric'])
                    data = stream.getvalue()
                elif path.endswith('.npy'):
                    np.save(stream, packet['numeric']['depth'])
                    data = stream.getvalue()
                else:
                    data = pcd_bytes(packet['numeric']['points'])
                self.send(data, 'application/octet-stream', filename=path.rsplit('/', 1)[1])
            elif path in ('/', '/app.js', '/style.css'):
                name = 'index.html' if path == '/' else path[1:]
                kind = {'index.html':'text/html; charset=utf-8', 'app.js':'text/javascript', 'style.css':'text/css'}[name]
                self.send((UI/name).read_bytes(), kind)
            else:
                self.send(b'{"error":"Not found"}', code=404)

        def do_POST(self):
            # Local UI commands only; reject cross-origin browser requests.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.send(b'{"error":"Origin rejected"}', code=403)
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 <= size <= 4096:
                    raise ValueError('Request too large')
                body = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(body, dict) or not isinstance(body.get('label', ''), str):
                    raise ValueError('Invalid request')
                path = urlsplit(self.path).path
                if path == '/api/source':
                    engine.select_source(body.get('device'), body.get('depth_enabled', False))
                    self.send(b'{"ok":true}')
                    return
                if path not in ('/api/start', '/api/stop', '/api/calibrate'):
                    raise ValueError('Unknown action')
                engine.command(path.rsplit('/', 1)[1], body.get('label', ''))
                self.send(b'{"ok":true}')
            except (ValueError, OSError) as exc:
                self.send(json.dumps({'error': str(exc)}).encode(), code=409)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--device', default=None, help='Optional startup camera; otherwise choose in the UI')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--fps', type=float, default=5)
    parser.add_argument('--border', type=float, default=.15)
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--model', type=Path, default=ROOT/'models/nnmini.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'recordings')
    args = parser.parse_args()
    if not .1 <= args.fps <= 60 or not 0 <= args.border < .45:
        parser.error('fps must be 0.1–60; border must be 0–0.45 (exclusive)')
    engine = Engine(args)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(engine))
    engine.thread.start()
    print(f'Tactile Recorder: http://127.0.0.1:{args.port} ({"DEMO" if args.demo else args.device or "choose a camera in the UI"})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == '__main__':
    main()
