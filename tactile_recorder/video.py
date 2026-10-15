"""Headless 3D preview and three-row recording, adapted from Daimon's recorder."""
import csv
from pathlib import Path
import cv2
import numpy as np

PANEL_WIDTH, IMAGE_HEIGHT, HEADER_HEIGHT = 480, 360, 28
ROW_HEIGHT = IMAGE_HEIGHT + HEADER_HEIGHT
SIZE = (PANEL_WIDTH * 2, ROW_HEIGHT * 3)
BACKGROUND = (22, 19, 16)
PANELS = (
    ('raw', 'Raw image', 0, 0, PANEL_WIDTH),
    ('deformation', 'Deformation image - RGB difference / gray = zero', 0, PANEL_WIDTH, PANEL_WIDTH),
    ('gradient', 'Image gradient', ROW_HEIGHT, 0, PANEL_WIDTH),
    ('hog', 'HOG', ROW_HEIGHT, PANEL_WIDTH, PANEL_WIDTH),
    ('deformation_3d', '3D deformation - estimated surface / Z x1', ROW_HEIGHT * 2, 0, SIZE[0]),
)


def render_surface(depth, width=960, height=360):
    """Fixed oblique view; native Z, full-surface fit, no GUI or browser required."""
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError('A finite 2D depth array is required for the 3D preview')
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
    yy, xx = np.indices(depth.shape, dtype=float)
    x = xx[::3, ::3].ravel() - (depth.shape[1]-1)/2
    y = yy[::3, ::3].ravel() - (depth.shape[0]-1)/2
    z = depth[::3, ::3].ravel()
    yaw, pitch = -.4, .8
    u = x*np.cos(yaw)-y*np.sin(yaw)
    v = x*np.sin(yaw)+y*np.cos(yaw)
    screen_y = v*np.cos(pitch)-z*np.sin(pitch)
    distance = v*np.sin(pitch)+z*np.cos(pitch)
    cx, cy = (u.min()+u.max())/2, (screen_y.min()+screen_y.max())/2
    scale = min(width*.88/max(320, np.ptp(u)), height*.78/max(180, np.ptp(screen_y)))
    coords = np.rint(np.column_stack((width/2+(u-cx)*scale, height/2+(screen_y-cy)*scale))).astype(int)
    limit = max(1., float(np.max(np.abs(depth))))
    colors = cv2.applyColorMap(np.clip((z/limit+1)*127.5, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    radius = max(1, int(round(scale)))
    for i in np.argsort(distance):
        cv2.circle(canvas, tuple(coords[i]), radius, tuple(int(c) for c in colors[i]), -1, cv2.LINE_AA)
    cv2.putText(canvas, f'X/Y: image pixels | Z: model units | color +/-{limit:.2f}',
                (16, height-14), cv2.FONT_HERSHEY_SIMPLEX, .43, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas


def compose(images, frame, elapsed, depth_limit=None, source_frame=None):
    """Same captured-frame images as the individual PNGs; preserve aspect ratio."""
    canvas = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, np.uint8)
    for key, title, top, left, width in PANELS:
        if key == 'raw':
            title += f' | #{frame} | {elapsed:.2f}s'
        elif key == 'deformation_3d' and source_frame is not None:
            title += f' | source #{source_frame}'
        cv2.putText(canvas, title, (left+10, top+20), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (230, 230, 230), 1, cv2.LINE_AA)
        image = images.get(key)
        if image is None:
            cv2.putText(canvas, 'Unavailable for this frame', (left+20, top+HEADER_HEIGHT+IMAGE_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (150, 150, 150), 1, cv2.LINE_AA)
            continue
        h, w = image.shape[:2]
        factor = min(width/w, IMAGE_HEIGHT/h)
        resized = cv2.resize(image, (max(1, round(w*factor)), max(1, round(h*factor))),
                             interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        h, w = resized.shape[:2]
        x, y = left+(width-w)//2, top+HEADER_HEIGHT+(IMAGE_HEIGHT-h)//2
        canvas[y:y+h, x:x+w] = resized
    return canvas


class StackedVideo:
    """Constant-rate movie with source-frame mapping and held frames across gaps."""
    def __init__(self, directory, fps):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video FPS must be positive')
        self.fps, self.frames = float(fps), 0
        self.started_ns = None
        self.last = self.last_source = None
        self.writer = None
        for filename, codec in (('overview.mp4', 'mp4v'), ('overview.avi', 'MJPG')):
            path = Path(directory) / filename
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, SIZE)
            if writer.isOpened():
                self.writer, self.path, self.codec = writer, path, codec
                break
            writer.release()
            path.unlink(missing_ok=True)
        if self.writer is None:
            raise OSError('Neither the MP4 nor AVI video encoder could be opened')
        try:
            self.index_file = (Path(directory)/'video_frames.csv').open('w', newline='')
        except OSError:
            self.writer.release()
            raise
        self.index = csv.writer(self.index_file)
        self.index.writerow(['video_frame', 'video_time_seconds', 'capture_frame', 'source_frame', 'capture_monotonic_ns', 'held'])
        self.index_file.flush()

    def elapsed(self, time_ns):
        return 0. if self.started_ns is None else max(0., (time_ns-self.started_ns)/1e9)

    def _write(self, canvas, source, held):
        self.writer.write(canvas)
        self.index.writerow([self.frames, self.frames/self.fps, *source, held])
        self.frames += 1

    def add(self, canvas, capture_frame, source_frame, time_ns):
        if self.writer is None:
            raise RuntimeError('Video is closed')
        if self.started_ns is None:
            self.started_ns = time_ns
        index = round(self.elapsed(time_ns)*self.fps)
        while self.frames < index:
            self._write(self.last, self.last_source, True)
        source = (capture_frame, source_frame, time_ns)
        written_index = None
        if self.frames == index:
            written_index = index
            self._write(canvas, source, False)
        self.last, self.last_source = canvas.copy(), source
        self.index_file.flush()
        return written_index

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.index_file.close()
