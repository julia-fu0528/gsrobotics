"""Lossless, frame-aligned session output with a CSV commit index."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import cv2
import numpy as np
from .features import pcd_bytes
from .video import StackedVideo, SIZE, compose, render_surface


class Recording:
    def __init__(self, output, label, metadata):
        self.path = Path(output) / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True)
        for name in ('raw', 'image', 'deformation', 'gradient', 'hog', 'depth_preview', 'deformation_3d', 'overview', 'numeric', 'depth', 'pcd'):
            (self.path / name).mkdir()
        self.count = 0
        self.started = datetime.now(timezone.utc).isoformat()
        self.metadata = dict(metadata, label=label, started_at=self.started, saved_frames=0,
                             status='recording', schema_version=2)
        self.video = StackedVideo(self.path, metadata.get('requested_hz', 5))
        self.metadata['video'] = dict(file=self.video.path.name, codec=self.video.codec,
                                     fps=self.video.fps, size=list(SIZE),
                                     rows=[['raw', 'deformation'], ['gradient', 'hog'], ['deformation_3d']],
                                     surface_view='Fixed yaw -0.4 rad, pitch 0.8 rad, Z x1, full-surface fit; independent of browser orbit.',
                                     timing='Monotonic capture timestamps; hold prior frame across gaps; video_frames.csv maps movie frames to captures. PNGs retain all captures.')
        self.file = (self.path / 'frames.csv').open('w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['frame', 'source_frame', 'capture_start_ns', 'capture_end_ns',
                              'monotonic_ns', 'reference_epoch', 'depth_saved', 'video_frame'])
        self._metadata()

    def _metadata(self):
        target = self.path / 'session.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(self.metadata, indent=2) + '\n')
        temp.replace(target)

    def save(self, packet):
        stem = f'{self.count:06d}'
        images = dict(packet['images'])
        depth = packet['numeric'].get('depth')
        if depth is not None:
            images['deformation_3d'] = render_surface(depth)
        overview = compose(images, self.count, self.video.elapsed(packet['monotonic_ns']),
                           packet['numeric'].get('depth_display_limit'), packet['source_frame'])
        images['overview'] = overview
        for key in ('raw', 'image', 'deformation', 'gradient', 'hog', 'depth_preview', 'deformation_3d', 'overview'):
            if key in images:
                if not cv2.imwrite(str(self.path / key / (stem+'.png')), images[key]):
                    raise OSError(f'Failed to write {key}')
        np.savez_compressed(self.path / 'numeric' / (stem+'.npz'), **packet['numeric'])
        if depth is not None:
            np.save(self.path / 'depth' / (stem+'.npy'), depth)
            (self.path / 'pcd' / (stem+'.pcd')).write_bytes(pcd_bytes(packet['numeric']['points']))
        video_frame = self.video.add(overview, self.count, packet['source_frame'], packet['monotonic_ns'])
        self.writer.writerow([self.count, packet['source_frame'], packet['capture_start_ns'],
                              packet['capture_end_ns'], packet['monotonic_ns'], packet['epoch'], depth is not None, video_frame])
        self.file.flush()
        self.count += 1

    def close(self, error=None):
        if self.file.closed:
            return
        self.file.close()
        self.video.close()
        self.metadata['video']['frames'] = self.video.frames
        self.metadata.update(saved_frames=self.count, ended_at=datetime.now(timezone.utc).isoformat(),
                             status='error' if error else 'complete', error=error)
        self._metadata()
