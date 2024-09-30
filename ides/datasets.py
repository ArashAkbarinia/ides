"""
Dataset helper!
"""

import numpy as np

from braindecode import augmentation as eeg_aug
import torch
from torch.utils import data as torch_data

import utils


def _average_every_n(arr, n):
    result = []
    for i in range(0, len(arr), n):
        chunk = arr[i:i + n]
        result.append(np.mean(chunk, axis=0))
    return np.array(result)


def clamp_outliers(arr, num_stds=20, axis=3):
    """
    Clamp outlier values in an array to be within num_stds standard deviations
    of the mean.

    Args:
        arr: The input array.
        num_stds: The number of standard deviations to consider an outlier.
        axis: The axis of operation.

    Returns:
        A new array with outliers clamped to num_stds standard deviations
        from the mean.
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    lower_bound = mean - num_stds * std
    upper_bound = mean + num_stds * std
    clamped_arr = np.clip(arr, lower_bound, upper_bound)
    return clamped_arr


class EEG2DatasetBase(torch_data.Dataset):
    def __init__(self, sfreq, subject, eeg_dir, split, features_file, single_trial=False,
                 inter_subjects=False, which_trials=None, which_imgs=None, augmentations=None,
                 avg_concepts=False, clamp_th=None, temporal_embedding=False):
        self.temporal_embedding = temporal_embedding
        self.avg_concepts = avg_concepts
        self.split = split
        if inter_subjects:
            subjects = np.arange(1, 11)
            subjects = np.delete(subjects, subject - 1)
        else:
            subjects = [subject]

        # for making the concept labels
        num_concepts = 200 if split == 'test' else 1654
        img_per_concept = 1 if split == 'test' else 10

        self.eeg_metadata = np.load(f"{eeg_dir}image_metadata.npy", allow_pickle=True)[()]
        concept_key = "test_img_concepts" if split == 'test' else "train_img_concepts"
        path_suffix = 'test' if split == 'test' else 'training'
        eeg_concepts = np.array([_c[6:] for _c in self.eeg_metadata[concept_key]])

        if which_imgs is not None:
            # negative images are excluded
            if np.sum(which_imgs) < 0:
                select_imgs = np.arange(img_per_concept)
                select_imgs = np.delete(select_imgs, np.abs(which_imgs))
            else:
                select_imgs = which_imgs
            new_img_per_concept = len(select_imgs)
            select_imgs = [
                _i for _s in select_imgs for _i in
                np.arange(_s, num_concepts * img_per_concept, img_per_concept)
            ]
            select_imgs = np.array(select_imgs)
            select_imgs.sort()
            img_per_concept = new_img_per_concept
            eeg_concepts = eeg_concepts[select_imgs]

        # parsing the sfreq
        sfreq, sfreq_suffix, resample_sfreq = utils.parse_sfreq(sfreq)
        if resample_sfreq is None:
            self.online_sfreq = None
        else:
            self.online_sfreq = np.round(np.linspace(0, sfreq - 1, resample_sfreq))

        all_eeg_signals = []
        all_features_ind = []
        all_labels = []
        all_concepts = []
        for subject in subjects:
            eeg_sub_dir = f"{eeg_dir}Preprocessed_data_{sfreq}Hz{sfreq_suffix}/sub-{subject:02d}/"
            eeg_path = f"{eeg_sub_dir}preprocessed_eeg_{path_suffix}.npy"
            eeg_signal = np.load(eeg_path, allow_pickle=True)['preprocessed_eeg_data']
            if clamp_th is not None:
                eeg_signal = clamp_outliers(eeg_signal, clamp_th)

            if which_imgs is not None:
                eeg_signal = eeg_signal[select_imgs]

            if which_trials is not None:
                # negative trials are excluded
                if np.sum(which_trials) < 0:
                    select_trials = np.arange(eeg_signal.shape[1])
                    select_trials = np.delete(select_trials, np.abs(which_trials))
                else:
                    select_trials = which_trials
                eeg_signal = eeg_signal[:, select_trials]

            if single_trial:
                num_trials = eeg_signal.shape[1]
                eeg_signal = np.reshape(
                    eeg_signal,
                    (eeg_signal.shape[0] * num_trials, eeg_signal.shape[2], eeg_signal.shape[3])
                )
            else:
                num_trials = 1
                eeg_signal = np.mean(eeg_signal, axis=1)
            # adding subject data to the entire list
            if which_imgs is not None:
                s_features_ind = select_imgs.repeat(num_trials)
            else:
                s_features_ind = np.arange(num_concepts * img_per_concept).repeat(num_trials)
            s_labels = np.arange(num_concepts).repeat(img_per_concept * num_trials)
            all_eeg_signals.append(eeg_signal)
            all_features_ind.append(s_features_ind)
            all_labels.append(s_labels)
            all_concepts.append(eeg_concepts.repeat(num_trials))
        # collapsing all subjects
        self.eeg_signal = np.concatenate(all_eeg_signals)
        self.features_ind = np.concatenate(all_features_ind)
        self.labels = np.concatenate(all_labels)
        self.eeg_concepts = np.concatenate(all_concepts)
        # reading the feature file
        self.features = np.load(features_file, allow_pickle=True)
        if which_imgs is None:
            self.template_features = _average_every_n(self.features, img_per_concept)
        else:
            self.template_features = _average_every_n(
                self.features[select_imgs], len(select_imgs) // num_concepts
            )

        self.transforms = _augmentation_transforms(augmentations, sfreq)

    def apply_transformations(self, x):
        """Applying transformations to the EEG signal."""
        for transform in self.transforms:
            x = transform(x)
        return x

    def get_eeg(self, index):
        eeg_signal = self.eeg_signal[index]
        if self.avg_concepts:
            same_concept = self.labels == self.labels[index]
            same_concept[index] = False
            same_concepts_inds = np.where(same_concept)[0]
            np.random.shuffle(same_concepts_inds)
            same_concepts_eeg = self.eeg_signal[same_concepts_inds[:7]]
            same_concepts_eeg = np.mean(same_concepts_eeg, axis=0)
            eeg_signal = (eeg_signal + same_concepts_eeg) / 2
        if self.online_sfreq is not None:
            if self.split == 'test':
                eeg_signal = eeg_signal[:, self.online_sfreq.astype('int')]
            else:
                eeg_signal = _resample_channel_wise(eeg_signal, self.online_sfreq)
        eeg_signal = torch.tensor(eeg_signal, dtype=torch.float)
        eeg_signal = self.apply_transformations(eeg_signal)
        if self.temporal_embedding:
            eeg_signal = eeg_signal.permute(1, 0)
        return eeg_signal

    def get_features(self, index):
        return torch.tensor(self.features[self.features_ind[index]], dtype=torch.float)


def _resample_channel_wise(eeg, intervals):
    num_rows, total_columns = eeg.shape
    num_columns = len(intervals)

    # Generate random column indices for each row
    random_columns = np.random.choice(total_columns, size=(num_rows, num_columns))

    # Sort the selected column indices along the columns axis
    random_columns.sort(axis=1)

    # Use fancy indexing to select the random columns from the input array
    resampled_eeg = eeg[np.arange(num_rows)[:, np.newaxis], random_columns]

    # resampled_eeg = np.zeros((eeg.shape[0], len(intervals)))
    # for ch in range(eeg.shape[0]):
    #     inds = np.array([
    #         *[np.random.randint(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)],
    #         intervals[-1]
    #     ])
    #     resampled_eeg[ch] = eeg[ch, inds.astype('int')]
    return resampled_eeg


def _resample_eeg(eeg, intervals):
    inds = np.array([
        *[np.random.randint(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)],
        intervals[-1]
    ])
    return eeg[:, inds.astype('int')]


def _augmentation_transforms(augmentations, sfreq):
    transforms = []
    if augmentations is not None:
        for aug in augmentations:
            # BandstopFilter SensorsRotation
            if aug == 'FrequencyShift':
                aug_transform = eeg_aug.FrequencyShift(0.5, sfreq)
            elif aug == 'FTSurrogate':
                aug_transform = eeg_aug.FTSurrogate(0.5)
            elif aug == 'ChannelsDropout':
                aug_transform = eeg_aug.ChannelsDropout(0.5)
            elif aug == 'GaussianNoise':
                aug_transform = eeg_aug.GaussianNoise(0.5, std=1.0)
            elif aug == 'SmoothTimeMask':
                aug_transform = eeg_aug.SmoothTimeMask(0.5, mask_len_samples=50)
            else:
                print(f"Augmentation {aug} is not supported.")
                continue
            transforms.append(aug_transform)
    return transforms
