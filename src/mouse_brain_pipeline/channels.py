"""Channel identifiers and their human-readable labels.

The two supplied folders are BOTH real biological signal channels:

  * ``green_signal``     -- the GREEN biological dye.
  * ``channel_2_signal`` -- the RED biological dye. It may be weaker / show fewer
                            visible cells because it was more photobleached during
                            imaging -- this is a property of the data, NOT a target
                            to tune detection towards.

Neither channel is a background / autofluorescence channel. The internal names
are kept stable for compatibility (directory keys, CSV ``channel`` column, config
sections); only the *displayed* label changes to "green/red signal channel".
"""

from __future__ import annotations

# Neutral internal channel names (the physical dye is encoded by the label below).
GREEN_SIGNAL = "green_signal"
CHANNEL_2_SIGNAL = "channel_2_signal"
BACKGROUND = "background"

# Human-readable labels for legends, titles, printed reports and metadata.
CHANNEL_DISPLAY_NAMES = {
    GREEN_SIGNAL: "green signal channel",
    CHANNEL_2_SIGNAL: "red signal channel",
    BACKGROUND: "background channel",
}


def channel_display_name(channel: str) -> str:
    """Human label for a channel id (e.g. ``channel_2_signal`` -> red signal channel)."""
    return CHANNEL_DISPLAY_NAMES.get(channel, str(channel))
