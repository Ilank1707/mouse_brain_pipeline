"""mouse_brain_pipeline.

Tools to audit, prepare, pilot-test and (eventually) analyse a very large
serial two-photon mouse-brain dataset consisting of two *signal* channels.

Design rules baked into this package:

* The two supplied folders are BOTH biological signal: ``green_signal`` is the
  GREEN dye and ``channel_2_signal`` is the RED dye. Neither is ever silently
  used as the anatomical/autofluorescence background channel required by the
  Cellfinder/Brainmapper classifier.
* Raw TIFFs are read-only. Nothing is renamed, moved, overwritten or recompressed.
* Nothing loads a whole brain into RAM -- reads are per-plane / per-tile and lazy.
* The built-in 3D blob detector is an EXPERIMENTAL pilot tool. Its outputs are
  "candidate detections", never final cell counts.
"""

__version__ = "0.1.0"

# Channel ids + human labels. green_signal -> "green signal channel",
# channel_2_signal -> "red signal channel". Internal names stay stable.
from .channels import (  # noqa: E402
    BACKGROUND,
    CHANNEL_2_SIGNAL,
    CHANNEL_DISPLAY_NAMES,
    GREEN_SIGNAL,
    channel_display_name,
)

__all__ = [
    "__version__", "GREEN_SIGNAL", "CHANNEL_2_SIGNAL", "BACKGROUND",
    "CHANNEL_DISPLAY_NAMES", "channel_display_name",
]
