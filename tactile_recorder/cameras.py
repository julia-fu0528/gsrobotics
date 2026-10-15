"""Enumerate Linux camera sources without opening or capturing from them."""
from pathlib import Path

DEMO_DEVICE = '__demo__'


def list_cameras(sys_root=Path('/sys/class/video4linux'), dev_root=Path('/dev')):
    sys_root, dev_root = Path(sys_root), Path(dev_root)
    aliases = {}
    for alias in sorted((dev_root/'v4l/by-id').glob('*')):
        try:
            aliases.setdefault(alias.resolve(), str(alias))
        except OSError:
            continue
    cameras = []
    for entry in sorted(sys_root.glob('video*'), key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else -1):
        try:
            # UVC exposes a companion metadata node (index 1), not a video stream.
            if (entry/'index').exists() and (entry/'index').read_text().strip() != '0':
                continue
            name = (entry/'name').read_text().strip()
        except OSError:
            continue  # A device may disappear during a USB refresh.
        node = dev_root / entry.name
        device = aliases.get(node.resolve(), str(node))
        cameras.append(dict(device=device, path=str(node), name=name,
                            label=f'{name} ({node})', gelsight='gelsight mini' in name.lower()))
    return cameras


def camera_for_device(device, cameras):
    if device is None:
        return None
    path = Path('/dev')/f'video{device}' if str(device).isdigit() else Path(device)
    for camera in cameras:
        if path.resolve() == Path(camera['path']).resolve() or str(device) == camera['device']:
            return camera
    return None
