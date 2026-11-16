"""Camera ordering and full-resolution RGB integration for runtime weather."""
import numpy as np
from PIL import Image

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


def prepare_triplet(raw):
    images, masks = [], []
    for name in CAMERAS:
        cam = raw['cameras'][name]
        rgb, mask = np.asarray(cam['rgb']), np.asarray(cam['weather_dynamic_mask'])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('invalid RGB')
        if mask.dtype != np.uint8 or mask.shape != rgb.shape[:2]:
            raise ValueError('invalid aligned dynamic mask')
        images.append(np.array(Image.fromarray(rgb).resize((384, 256), Image.Resampling.BICUBIC)))
        masks.append(np.array(Image.fromarray(mask).resize((384, 256), Image.Resampling.NEAREST)))
    return np.stack(images), np.stack(masks)


def integrate_triplet(raw, weather):
    weather = np.asarray(weather)
    if weather.shape != (3, 256, 384, 3) or weather.dtype != np.uint8:
        raise ValueError('invalid weather triplet')
    # Restore original sensor dimensions; K, depth and actor geometry stay aligned.
    # Explicitly protect visible actors instead of relying on model conditioning.
    cameras = {}
    for i, name in enumerate(CAMERAS):
        camera = dict(raw['cameras'][name])
        original = camera['rgb']
        image = np.array(Image.fromarray(weather[i]).resize(
            (original.shape[1], original.shape[0]), Image.Resampling.BICUBIC))
        actor = camera['weather_dynamic_mask'] > 0
        image[actor] = original[actor]
        camera['rgb'] = image
        cameras[name] = camera
    return dict(raw, cameras=cameras)
