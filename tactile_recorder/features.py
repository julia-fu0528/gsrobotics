"""Image features; HOG adapted from Daimon_Robotics/tactile_recorder/image_features.py."""
import cv2
import numpy as np


def hog(gray, cell=8, bins=9):
    """Central differences, spatial/orientation interpolation, 2x2 L2-Hys blocks."""
    h, w = gray.shape
    rows, cols = h // cell, w // cell
    if min(rows, cols) < 2:
        raise ValueError('HOG requires at least 2x2 cells')
    # Same central-difference stencil/spatial voting as daimon/tactile/features.js.
    yy, xx = np.mgrid[1:rows*cell-1, 1:cols*cell-1]
    gx = gray[yy, xx+1] - gray[yy, xx-1]
    gy = gray[yy+1, xx] - gray[yy-1, xx]
    strength = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi) * bins / np.pi
    lo = np.floor(angle).astype(int)
    frac = angle - lo
    cx, cy = (xx+.5)/cell-.5, (yy+.5)/cell-.5
    ix, iy = np.floor(cx).astype(int), np.floor(cy).astype(int)
    hist = np.zeros((rows, cols, bins))
    for oy in (0, 1):
        for ox in (0, 1):
            x, y = ix+ox, iy+oy
            good = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
            weights = strength * (cx-ix if ox else 1-cx+ix) * (cy-iy if oy else 1-cy+iy)
            np.add.at(hist, (y[good], x[good], lo[good] % bins), (weights*(1-frac))[good])
            np.add.at(hist, (y[good], x[good], (lo[good]+1) % bins), (weights*frac)[good])
    blocks = np.stack([np.stack([hist[y:y+2, x:x+2] for x in range(cols-1)]) for y in range(rows-1)])
    norm = np.sqrt(np.sum(blocks**2, axis=(2,3,4), keepdims=True)+1e-10)
    blocks = np.minimum(blocks/norm, .2)
    blocks /= np.sqrt(np.sum(blocks**2, axis=(2,3,4), keepdims=True)+1e-10)
    visual = np.zeros_like(hist)
    count = np.zeros((rows,cols,1))
    for oy in (0,1):
        for ox in (0,1):
            visual[oy:oy+rows-1,ox:ox+cols-1] += blocks[:,:,oy,ox]
            count[oy:oy+rows-1,ox:ox+cols-1] += 1
    return dict(image_hog_histograms=hist, image_hog_blocks=blocks,
                image_hog_descriptor=blocks.ravel(), image_hog_cells=visual/np.maximum(count,1),
                image_hog_bin_angles_rad=np.arange(bins)*np.pi/bins)



class ImageFeatures:
    """Frozen two-frame reference, with overlays on the actual sensing image."""
    def __init__(self):
        self.references = []
        self.color_references = []

    def update(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        if len(self.references) < 2:
            self.references.append(gray.copy())
            self.color_references.append(bgr.astype(np.float32) / 255)
        reference = np.mean(self.references, axis=0)
        color_reference = np.mean(self.color_references, axis=0)
        delta = gray - reference
        gy, gx = np.gradient(delta)
        numeric = dict(image_gray=gray, image_reference_gray=reference,
                       image_reference_bgr=color_reference,
                       image_delta_bgr=bgr.astype(np.float32)/255-color_reference,
                       image_gray_delta=delta, image_gradient_x=gx, image_gradient_y=gy,
                       image_gradient_magnitude=np.hypot(gx, gy), **hog(delta))
        arrows, glyphs = bgr.copy(), (bgr * .45).astype(np.uint8)
        for top in range(0, gray.shape[0], 12):
            for left in range(0, gray.shape[1], 12):
                # Keep thin edges visible even when they fall between grid centers.
                tile = numeric['image_gradient_magnitude'][top:top+12, left:left+12]
                dy, dx = np.unravel_index(np.argmax(tile), tile.shape)
                y, x = top + dy, left + dx
                d = np.array([gx[y, x], gy[y, x]]) * 160
                length = np.linalg.norm(d)
                if length < .8:
                    continue
                end = tuple(np.rint([x, y] + d * min(1, 9/length)).astype(int))
                cv2.arrowedLine(arrows, (x, y), end, (0, 0, 0), 2, cv2.LINE_AA, tipLength=.3)
                cv2.arrowedLine(arrows, (x, y), end, (0, 255, 255), 1, cv2.LINE_AA, tipLength=.3)
        for y, x, b in np.argwhere(numeric['image_hog_cells'] > .008):
            weight = numeric['image_hog_cells'][y, x, b]
            center = np.array([(x+.5)*8, (y+.5)*8])
            d = 3.2*min(1, weight*2)*np.array([np.cos(b*np.pi/9), np.sin(b*np.pi/9)])
            cv2.line(glyphs, tuple(np.rint(center-d).astype(int)),
                     tuple(np.rint(center+d).astype(int)), (100, 255, 100), 1, cv2.LINE_AA)
        return numeric, arrows, glyphs


def points_from_depth(depth):
    """Organized XYZ: x = column, y = row, z = signed model depth; pixel/model units."""
    y, x = np.indices(depth.shape, dtype=np.float32)
    return np.stack((x, y, depth), axis=-1).astype(np.float32)


def pcd_bytes(points):
    points = np.ascontiguousarray(points.reshape(-1, 3), dtype='<f4')
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n'
              f'COUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
              f'POINTS {len(points)}\nDATA binary\n')
    return header.encode('ascii') + points.tobytes()
