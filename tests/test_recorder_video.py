"""Decode saved videos to check layout, timing, and correspondence with PNGs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from tactile_recorder.features import ImageFeatures
from tactile_recorder.video import StackedVideo, compose, render_surface, SIZE, ROW_HEIGHT, HEADER_HEIGHT
from tactile_recorder.recording import Recording
from test_tactile_recorder import Engine, args, packet


class VideoTest(unittest.TestCase):
    def test_five_panel_three_row_layout(self):
        colors = {'raw':(30,40,50), 'deformation':(60,70,80), 'gradient':(90,100,110),
                  'hog':(120,130,140), 'deformation_3d':(150,160,170)}
        images = {key:np.full((360,960 if key=='deformation_3d' else 480,3), color, np.uint8)
                  for key,color in colors.items()}
        canvas = compose(images, 7, 1.2)
        self.assertEqual(canvas.shape, (SIZE[1],SIZE[0],3))
        for key,x,y in [('raw',240,208),('deformation',720,208),('gradient',240,ROW_HEIGHT+208),
                        ('hog',720,ROW_HEIGHT+208),('deformation_3d',480,ROW_HEIGHT*2+208)]:
            np.testing.assert_array_equal(canvas[y,x],colors[key])

    def test_missing_depth_has_placeholder_not_stale_surface(self):
        current=compose({'raw':np.full((240,320,3),200,np.uint8)},1,0)
        missing=current[ROW_HEIGHT*2+HEADER_HEIGHT:]
        self.assertGreater(np.unique(missing.reshape(-1,3),axis=0).shape[0],1)
        self.assertLess(missing.mean(),40)

    def test_surface_render_is_deterministic_without_mutating_depth(self):
        y,x=np.mgrid[:240,:320]
        depth=(-5*np.exp(-((x-160)**2+(y-120)**2)/1000)).astype(np.float32)
        before=depth.copy()
        a=render_surface(depth)
        np.testing.assert_array_equal(depth,before)
        np.testing.assert_array_equal(a,render_surface(depth))
        self.assertEqual(a.shape,(360,960,3))
        self.assertGreater(np.unique(a.reshape(-1,3),axis=0).shape[0],100)
        with self.assertRaises(ValueError):
            render_surface(np.full((10,10),np.nan))

    def test_encoded_video_holds_frames_across_capture_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            video=StackedVideo(temp,5)
            a=np.full((SIZE[1],SIZE[0],3),40,np.uint8)
            b=np.full_like(a,200)
            self.assertEqual(video.add(a,0,12,1_000_000_000),0)
            self.assertEqual(video.add(b,1,13,1_600_000_000),3)
            video.close()
            capture=cv2.VideoCapture(str(video.path))
            decoded=[]
            while True:
                ok,frame=capture.read()
                if not ok: break
                decoded.append(frame)
            capture.release()
            self.assertEqual(len(decoded),4)
            self.assertEqual(decoded[0].shape,(SIZE[1],SIZE[0],3))
            for frame in decoded[:3]:self.assertLess(abs(frame.mean()-40),6)
            self.assertLess(abs(decoded[3].mean()-200),6)
            with (Path(temp)/'video_frames.csv').open() as file:rows=list(csv.DictReader(file))
            self.assertEqual([row['source_frame'] for row in rows],['12','12','12','13'])
            self.assertEqual([row['held'] for row in rows],['False','True','True','False'])

    def test_codec_fallback_and_failure(self):
        class Writer:
            def __init__(self,opened):self.opened=opened;self.released=False
            def isOpened(self):return self.opened
            def release(self):self.released=True
        with tempfile.TemporaryDirectory() as temp:
            mp4,avi=Writer(False),Writer(True)
            with patch('tactile_recorder.video.cv2.VideoWriter',side_effect=[mp4,avi]):
                video=StackedVideo(temp,5)
                self.assertEqual(video.path.name,'overview.avi')
                self.assertTrue(mp4.released)
                video.close()
                self.assertTrue(avi.released)
            with patch('tactile_recorder.video.cv2.VideoWriter',side_effect=[Writer(False),Writer(False)]):
                with self.assertRaises(OSError):StackedVideo(temp,5)

    def test_color_difference_preserves_sign_and_frozen_reference(self):
        features=ImageFeatures()
        base=np.full((24,32,3),100,np.uint8)
        features.update(base);features.update(base)
        contact=base.copy();contact[4:12,4:12]=[150,100,50]
        numeric,_,_=features.update(contact)
        np.testing.assert_allclose(numeric['image_delta_bgr'][5,5],[50/255,0,-50/255],atol=1e-7)
        np.testing.assert_allclose(numeric['image_reference_bgr'],100/255,atol=1e-7)

    def test_session_pngs_and_video_share_same_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            engine=Engine(args(temp))
            packet(engine)
            frame=packet(engine)
            recording=Recording(temp,'layout-test',engine.metadata())
            recording.save(frame);recording.close()
            path=recording.path
            for name in ('raw','deformation','gradient','hog','deformation_3d','overview'):
                self.assertTrue((path/name/'000000.png').is_file(),name)
            np.testing.assert_array_equal(cv2.imread(str(path/'deformation/000000.png')),frame['images']['deformation'])
            overview=cv2.imread(str(path/'overview/000000.png'))
            surface=cv2.imread(str(path/'deformation_3d/000000.png'))
            np.testing.assert_array_equal(overview[ROW_HEIGHT*2+HEADER_HEIGHT:],surface)
            metadata=json.loads((path/'session.json').read_text())
            self.assertEqual(metadata['video']['rows'],[['raw','deformation'],['gradient','hog'],['deformation_3d']])
            self.assertEqual(metadata['video']['frames'],1)
            capture=cv2.VideoCapture(str(path/metadata['video']['file']))
            ok,movie=capture.read();capture.release()
            self.assertTrue(ok)
            self.assertLess(np.abs(movie.astype(float)-overview).mean(),10)
            with np.load(path/'numeric/000000.npz') as numeric:
                self.assertIn('image_delta_bgr',numeric)


if __name__=='__main__':unittest.main()
