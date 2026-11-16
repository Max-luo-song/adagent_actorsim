"""History-only adapter for the existing three-frame TV model.

The model may mix its whole input window. Only its *last* camera triplet is
emitted at the current timestamp. Earlier outputs are never retroactively edited.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Frame:
    index: int
    timestamp: float
    epoch: str
    rgb: np.ndarray
    mask: np.ndarray


class CausalWeatherBuffer:
    def __init__(self): self._history=[]

    def process(self, frame, timestamp, epoch, rgb, mask, predict):
        if type(frame) is not int or frame<0 or not np.isfinite(timestamp):
            raise ValueError('invalid frame/time')
        rgb=np.asarray(rgb);mask=np.asarray(mask)
        if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[0]!=3 or rgb.shape[-1]!=3:
            raise ValueError('expected three uint8 RGB views')
        if mask.dtype!=np.uint8 or mask.shape!=rgb.shape[:3]:
            raise ValueError('expected three aligned uint8 masks')
        history=self._history if self._history and self._history[-1].epoch==epoch else []
        if history and (timestamp<=history[-1].timestamp or frame<=history[-1].index):
            raise ValueError('non-monotone history')
        current=Frame(frame,float(timestamp),epoch,rgb.copy(),mask.copy())
        candidate=(history+[current])[-3:]
        window=[candidate[0]]*(3-len(candidate))+candidate
        window_rgb=np.concatenate([f.rgb for f in window])
        window_mask=np.concatenate([f.mask for f in window])
        output=np.asarray(predict(window_rgb,window_mask))
        if output.shape!=window_rgb.shape or output.dtype!=np.uint8:
            raise ValueError('invalid model result')
        # Commit only after successful inference and validation.
        self._history=candidate
        return output[-3:].copy(),dict(current_frame=frame,current_timestamp=float(timestamp),
            epoch=epoch,window_frames=[f.index for f in window],
            window_timestamps=[f.timestamp for f in window],
            history_padding=3-len(candidate),emitted_camera_indices=[6,7,8],
            future_frames_used=False)
