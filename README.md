# py_algo
Some algorithms I share


## Color correction
- modified from [sd-webui-stablesr colorfix](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
- supports cpu, cuda(fp16/fp32)


```python
import numpy as np
import torch
from torch import Tensor
from color_correction import ColorCorrection

in_tensor: Tensor
ref_tensor: Tensor

color_correction = ColorCorrection()
color_correction = color_correction.to(device=device)
with torch.inference_mode():
    d_out: Tensor = color_correction(in_tensor, ref_tensor)
    d_out = torch.clamp(d_out, 0., 1.)

```


# DarkerThinnerLines

PyTorch module that makes lines darker and thinner.

The input tensor must be a float tensor (`float32` or `float16`).

## Usage

```python
from darker_thinner_lines import DarkerThinnerLines

model = DarkerThinnerLines(darken=0.6, thin=0.4)
output = model(image)  # image: float tensor, values in [0, 1]
```

## Arguments

| Argument | Default | Help |
|---|---|---|
| `darken` | `0.6` | Darkening strength. Set to `0` to disable. Recommended: `0.1` to `2.0`. |
| `thin` | `0.4` | Line thinning strength. Set to `0` to disable. High values may create artifacts around lines. Recommended: `0` to `0.8`. |
| `luma_max` | `0.75` | Pixels brighter than this luma (Y) value are not darkened. Recommended: `0.5` to `1.0`. |
| `min_change` | `0.016` | Pixels whose luma would change by less than this value are left untouched. Recommended: `0.01` to `0.03`. |
| `luma_min` | `0.02` | Pixels darker than this luma (Y) value are not darkened further. Recommended: `0` to `0.20`. |


## Video script

Apply the filter to a video file.

```bash
python ./tests/darker_thinner_video.py -i in.mkv -o out.mkv --darken 0.6 --thin 0.3
```

### Arguments

| Argument | Default | Help |
|---|---|---|
| `--input`, `-i` | `in.mkv` | Input video file. |
| `--darken` | `0.6` | Darkening strength. Set to `0` to disable. |
| `--thin` | `0.3` | Line thinning strength. Set to `0` to disable. High values may create artifacts. |
| `--output`, `-o` | *(empty)* | Output video file. If empty, the name is generated from the input file name. |
| `--out_dir` | *(empty)* | Output directory. If empty, the input file's directory is used. |
| `--append_params` | `False` | Append the parameters (`darken`, `thin`) to the output file name. |
| `--cpu` | `False` | Run on CPU instead of GPU (slower). |
| `--debug` | `False` | Enable verbose output for debugging. |

