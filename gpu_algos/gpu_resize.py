from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor

from denc import NnFrame



def gpu_resize_(
    frame: NnFrame,
    scale_factor: Optional[float | int | list[float]] = None,
    out_size: tuple[int, int] | None = None,
    interpolation_method: str = "bicubic",
) -> NnFrame:
    in_x: Tensor = frame.tensor
    _out_size: tuple[int, int] = out_size
    if isinstance(scale_factor, float | int):
        _out_size = tuple(
            [int(((x * scale_factor) // 2 ) * 2) for x in in_x.shape[2:]]
        )

    in_x = in_x.to(dtype=torch.float16)
    out_x: Tensor = F.interpolate(
        input=in_x,
        size=_out_size,
        mode=interpolation_method,
        align_corners=False,
        antialias=bool(interpolation_method in ('bilinear', 'bicubic'))
    )
    frame.tensor = torch.clamp(out_x.contiguous(), 0, 1.0)

    return frame



def gpu_resize_to_(
    frame: NnFrame,
    out_size: tuple[int, int],
    interpolation_method: str = "bicubic",
) -> NnFrame:
    """outsize: (h, w)"""
    if out_size is None:
        raise ValueError(f"gpu_resize_to_: output size must not be None")
    return gpu_resize_(
        frame,
        out_size=out_size,
        interpolation_method=interpolation_method
    )
