# Experiment: AdaIN alpha=0.3

- **Status:** FAIL
- **Time:** 1186s (19.8 min)
- **Date:** 2026-03-23 02:26
- **Save dir:** `./visualizations2/adain_alpha0.3`

## Details
- Style transfer alpha: 0.3
- Higher alpha = more synthetic style

- **Output files:** 0

## Last 20 lines of output
```
Computing synthetic image statistics...
  Computed from 200 images

Loading real photos from ./data/real_photos...
  Found 27 photos

Applying AdaIN style transfer (alpha=0.3)...

Running inference (1 image at a time)...
  [10/27]
Traceback (most recent call last):
  File "D:\DL\run_adain.py", line 312, in <module>
    main()
  File "D:\DL\run_adain.py", line 259, in main
    pred_orig = model(img_orig).cpu()
                ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```
