# Synthetic fixtures

`images/01.png`, `02.png`, and `10.PNG` are 32 × 24 constant BGR images
with values 20, 60, and 100. `synthetic.avi` contains four 32 × 24 MJPEG
frames with values 20, 60, 100, and 140 at 5 FPS. All are generated from
scratch with NumPy and OpenCV; no private or public media
is used. Encoder versions may change binary bytes. Decoded content is tested.
The text file in `images/` deliberately tests extension filtering.
