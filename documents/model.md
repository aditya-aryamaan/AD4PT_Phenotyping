# Model provenance

The default detector uses a separately obtained YOLO11s checkpoint. Model weights
are excluded from Git. The application verifies the configured SHA256 before
loading the checkpoint and checks its class labels.

| Field | Value |
| --- | --- |
| Publisher | `shivamjha98/yolo11s-strawberry-detection` on Hugging Face |
| Revision | `4c2fbec408c569063751b124f6f36e6ed313ad61` |
| File | `strawberry_yolo_best.pt` |
| Size recorded during inspection | 19,176,218 bytes |
| SHA256 | `566a60e937bd7cbfffce6fd050e1d53bdda5119d01a31a35c54efd6653378b48` |
| Class IDs | `0: strawberry-ripe`, `1: strawberry-unripe` |
| Framework recorded in checkpoint | Ultralytics `8.4.14` |
| Local inspection date | 2026-09-16 |

[Upstream files at the pinned revision](https://huggingface.co/shivamjha98/yolo11s-strawberry-detection/tree/4c2fbec408c569063751b124f6f36e6ed313ad61).
Download instructions are in the [README](../README.md#2-obtain-the-model).

The original local inspection recorded conflicting licensing information: the
model card declared MIT, while checkpoint metadata recorded AGPL-3.0 and an
Ultralytics license URL. These notes do not resolve that conflict. Check the
applicable upstream terms before using or redistributing the artifact. The
project's eventual code license will not change third-party terms.

The publisher's training-data description and reported metrics have not been
independently verified by this project. They are not AD4PT evaluation results.
Local smoke images and artificial-fruit tests do not establish held-out real-field
accuracy. This is a two-class model; it has no separate semi-ripe class.

To use a different trusted local checkpoint, create a detector JSON with its path,
SHA256, identifier, and ordered class list. The live counting CLI expects exactly
the ripe/unripe class pair. Paths in model JSON files are relative to those files.
The code currently supports `cpu` and `mps` devices.
