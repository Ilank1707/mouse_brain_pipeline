"""mouse_brain_pipeline.

Tools to audit, prepare, pilot-test and (eventually) analyse a very large
serial two-photon mouse-brain dataset consisting of two *signal* channels.

Design rules baked into this package:

* The two supplied folders are BOTH biological signal. Neither is ever silently
  used as the anatomical/autofluorescence background channel required by the
  Cellfinder/Brainmapper classifier.
* Raw TIFFs are read-only. Nothing is renamed, moved, overwritten or recompressed.
* Nothing loads a whole brain into RAM -- reads are per-plane / per-tile and lazy.
* The built-in 3D blob detector is an EXPERIMENTAL pilot tool. Its outputs are
  "candidate detections", never final cell counts.
"""

__version__ = "0.1.0"

# Neutral channel names used throughout -- the real biological markers are unknown.
GREEN_SIGNAL = "green_signal"
CHANNEL_2_SIGNAL = "channel_2_signal"
BACKGROUND = "background"

__all__ = ["__version__", "GREEN_SIGNAL", "CHANNEL_2_SIGNAL", "BACKGROUND"]
