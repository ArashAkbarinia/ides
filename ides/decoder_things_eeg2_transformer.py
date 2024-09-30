"""
The source code for an experiment in which we hypothesised that the single trial decoding
can help the decoding power of EGG signal.
"""

import numpy as np
import argparse
import datetime
import random
import time

import torch
import torch.nn as nn
from torch.utils import data as torch_data

import utils
import datasets

parser = argparse.ArgumentParser(description='Transformer experiment for EEG decoding.')
parser.add_argument(
    "--concept_loss", action=argparse.BooleanOptionalAction, default=False,
    help="if true the concept cross entropy loss is added to the loss function",
)
parser.add_argument(
    "--eeg_loss", nargs='+', type=str, default=["cosine"],
    choices=["cosine", "mse"],
    help="the list of EGG losses"
)


class EEG2Dataset(datasets.EEG2DatasetBase):
    def __init__(self, *args, **kwargs):
        super(EEG2Dataset, self).__init__(*args, **kwargs)
        print(f"EEG2 Dataset [{self.split}] successfully loaded with {self.__len__()} trials!")

    def __len__(self):
        return len(self.eeg_signal)

    def __getitem__(self, index):
        eeg_signal = self.get_eeg(index)
        features = self.get_features(index)
        return eeg_signal, features, self.labels[index]


class Trainer(utils.Trainer):
    """The trainer class."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.concept_loss = args.concept_loss
        self.concept_weight = 0.1
        self.mse_weight = 0.05
        self.eeg_loss = args.eeg_loss

        self.criterion_mse = nn.MSELoss(reduction='sum').to(args.device)

        print('Transformer experiment is ready!')

    def _one_epoch(self, loader, training=True):
        losses = {'eeg': [], 'img': [], 'mse': [], 'concept': [], 'total': []}
        if self.concept_loss is False or training is False:
            del losses['concept']
        if training is False or 'mse' not in self.eeg_loss:
            del losses['mse']
        predictions = []
        gts = []
        tops = {1: [], 3: [], 5: []}
        template_features_from_img = loader.dataset.template_features
        template_features_from_img = torch.tensor(template_features_from_img, dtype=torch.float)
        template_features_from_img = template_features_from_img.to(self.device)

        with torch.set_grad_enabled(training):
            for i, batch_data in enumerate(loader):
                data_len = len(batch_data[0])
                # EEG data
                eeg_signal = batch_data[0].to(self.device)
                # Features (e.g., like CLIP)
                features_from_img = batch_data[1].to(self.device)
                features_from_img = features_from_img / features_from_img.norm(dim=1, keepdim=True)
                features_target = torch.arange(data_len).to(self.device)
                # Concept labels
                concepts_target = batch_data[2].to(self.device)

                # project the features to a multimodal embedding space
                model_out = self.eeg2clip(eeg_signal)
                if type(model_out) is tuple:
                    class_prob, features_from_eeg = model_out
                else:
                    class_prob, features_from_eeg = model_out, model_out
                features_from_eeg = features_from_eeg / features_from_eeg.norm(dim=1, keepdim=True)
                # cosine similarity as the logits
                logits_eeg = self.logit_scale.exp() * features_from_eeg @ features_from_img.t()
                logits_img = logits_eeg.t()
                loss_eeg_cosine = self.criterion_cls(logits_eeg, features_target)
                loss_img_cosine = self.criterion_cls(logits_img, features_target)

                losses['eeg'].extend([loss_eeg_cosine.item()] * data_len)
                losses['img'].extend([loss_img_cosine.item()] * data_len)

                if 'mse' in losses:
                    loss_mse = self.criterion_mse(features_from_eeg, features_from_img)
                    losses['mse'].extend([loss_mse.item()] * data_len)
                    loss_feature = [loss_mse * self.mse_weight]
                else:
                    loss_feature = []

                if 'cosine' in self.eeg_loss:
                    loss_feature.extend([loss_eeg_cosine, loss_img_cosine])

                loss = torch.mean(torch.stack(loss_feature))
                if 'concept' in losses:
                    loss_concept = self.criterion_cls(class_prob, concepts_target)
                    losses['concept'].extend([loss_concept.item()] * data_len)
                    loss = loss + loss_concept * self.concept_weight

                losses['total'].extend([loss.item()] * data_len)

                # Test labels
                predictions, gts, tops = utils.report_topks(
                    features_from_eeg, template_features_from_img, concepts_target,
                    predictions, gts, tops
                )

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        losses = {key: np.mean(val) for key, val in losses.items()}
        test_results = {'predictions': predictions, 'gts': gts, 'tops': tops}
        return losses, test_results

    def test(self, db_loader, save_print=False):
        self.eeg2clip.eval()
        losses, test_results = self._one_epoch(db_loader, training=False)
        if save_print:
            self._save_test_predictions(test_results)
        tops = {f"top{key}": np.mean(val) for key, val in test_results['tops'].items()}
        return {'loss': losses['total']}, tops

    def get_test_loader(self, single_trial=False, test_set="test_out_distribution"):
        """Loading the test data."""
        test_features_file = f"{self.feature_extractor}{test_set}.npy"
        test_dataset = EEG2Dataset(
            self.sfreq, self.subject, self.eeg_dir, 'test', test_features_file, single_trial,
            clamp_th=self.clamp_th, temporal_embedding=self.temporal_embedding
        )
        test_loader = torch_data.DataLoader(
            test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        return test_loader

    def train(self):
        """Training the network."""
        train_features_file = f"{self.feature_extractor}train.npy"
        train_exclude_trials = None if self.val_trials is None else np.array(self.val_trials) * -1
        train_exclude_imgs = None if self.val_imgs is None else np.array(self.val_imgs) * -1
        if self.val_trials is not None or self.val_imgs is not None:
            val_dataset = EEG2Dataset(
                self.sfreq, self.subject, self.eeg_dir, 'validation', train_features_file,
                False, self.inter_subjects, self.val_trials, self.val_imgs,
                clamp_th=self.clamp_th, temporal_embedding=self.temporal_embedding
            )
            val_loader = torch_data.DataLoader(
                val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers
            )

        train_dataset = EEG2Dataset(
            self.sfreq, self.subject, self.eeg_dir, 'training',
            train_features_file, self.single_trial, self.inter_subjects, train_exclude_trials,
            train_exclude_imgs, self.augmentations, self.avg_concepts, self.clamp_th,
            self.temporal_embedding
        )
        train_loader = torch_data.DataLoader(
            train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        test_loader = self.get_test_loader()

        best_test_loss = np.inf
        best_val_loss = np.inf
        reports = dict()
        for e in range(self.epochs):
            self.eeg2clip.train()
            loss_train, tops_train = self._one_epoch(train_loader)
            reports = utils.add_to_dict(loss_train, reports)
            reports = utils.add_to_dict({'train_top5': np.mean(tops_train['tops'][5])}, reports)

            if self.val_trials is not None or self.val_imgs is not None:
                loss_val, tops_val = self._one_epoch(val_loader, training=False)
                reports = utils.add_to_dict(loss_val, reports, prefix='val_')
                reports = utils.add_to_dict({'val_top5': np.mean(tops_val['tops'][5])}, reports)
            if self.scheduler is not None:
                if type(self.scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(loss_val['total'])
                else:
                    self.scheduler.step()

            self._save_network("eeg2clip_ck")
            if loss_val['total'] <= best_val_loss:
                best_val_loss = loss_val['total']
                self._save_network("eeg2clip_val")

            loss_test, tops_test = self.test(test_loader, save_print=False)
            reports = utils.add_to_dict(loss_test, reports, prefix='test_')
            reports = utils.add_to_dict(tops_test, reports)

            if loss_test['loss'] <= best_test_loss:
                best_test_loss = loss_test['loss']
                self._save_network("eeg2clip_test")

            if e % self.print_frequency == 0:
                loss_str = " ".join(
                    f"{_key} [{_val[-1]:.02f}]" for _key, _val in reports.items()
                )
                print(f"[{e:05d}/{self.epochs:05d}] losses: {loss_str}")
                self._save_losses(reports)
        _ = self.test(test_loader, save_print=True)


def main():
    """Training for one subject."""
    args = utils.parse_arguments(parser)

    starttime = datetime.datetime.now()
    seed_n = np.random.randint(args.seed)

    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    print('Subject %d' % args.subject)
    sub_trainer = Trainer(args)

    if args.test is not None:
        test_loader = sub_trainer.get_test_loader(args.single_trial, args.test_set)
        sub_trainer.test(test_loader, save_print=True)
        return

    sub_trainer.train()

    endtime = datetime.datetime.now()
    args.duration = str(endtime - starttime)
    print(f"subject {args.subject} duration: {args.duration}")
    utils.save_arguments(args)


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
