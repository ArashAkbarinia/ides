"""
Different EEG architectures.
"""

from braindecode import models as brain_models


class EEGNetv4(brain_models.EEGNetv4):
    def __init__(self, representation_size, channels_size, temporal_size):
        super().__init__(
            n_chans=channels_size, n_times=temporal_size, n_outputs=representation_size
        )
        self.channels_size = channels_size
        self.temporal_size = temporal_size
        self.representation_size = representation_size

    def save_params(self):
        """The required parameters to load the network."""
        return {
            "channels_size": self.channels_size,
            "temporal_size": self.temporal_size,
            "representation_size": self.representation_size,
        }
