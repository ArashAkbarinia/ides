"""
List of utility functions.
"""

import numpy as np
import pandas as pd
import os
import itertools
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils import data as torch_data

from eeg_transformer import EEGTransformer
from nice_network import EEGProjector as NiceNet
from atm_network import ATM_S_reconstruction_scale as ATMNet
from models import EEGNetv4


def save_arguments(args):
    json_file_name = os.path.join(args.out_dir, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_sfreq(sfreq):
    sfreq_parsed = sfreq.split('_')
    sfreq = int(sfreq_parsed[0])
    sfreq_suffix = ""
    resample_sfreq = None
    if len(sfreq_parsed) > 1:
        if "resample" not in sfreq_parsed[1]:
            sfreq_suffix = f"_{sfreq_parsed[1]}"
            if len(sfreq_parsed) > 2:
                resample_sfreq = int(sfreq_parsed[2].replace('resample', ''))
        else:
            resample_sfreq = int(sfreq_parsed[1].replace('resample', ''))
    return sfreq, sfreq_suffix, resample_sfreq


def load_model(args):
    sfreq, _, resample_sfreq = parse_sfreq(args.sfreq)
    if resample_sfreq is not None:
        sfreq = resample_sfreq
    state_dict = None
    if args.test is not None or args.resume is not None:
        filepath = args.resume if args.resume is not None else args.test
        checkpoint = torch.load(filepath, map_location='cpu')
        state_dict = checkpoint['state_dict']
        net_name = checkpoint['net_name'] if 'net_name' in checkpoint else args.arch
        if 'architecture' not in checkpoint['architecture'] or len(checkpoint['architecture']) == 0:
            if net_name in ['atm', 'nice']:
                arch_params = {
                    'representation_size': args.representation_size
                }
        else:
            arch_params = checkpoint['architecture']
            arch_params['dropout'] = args.dropout
        if 'channels_size' not in arch_params:
            arch_params['channels_size'] = 63
            arch_params['temporal_size'] = sfreq
    else:
        net_name = args.arch
        if net_name == 'transformer':
            arch_params = {
                'channels_size': sfreq if args.temporal_embedding else 63,
                'temporal_size': 63 if args.temporal_embedding else sfreq,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'hidden_dim': args.hidden_dim,
                'representation_size': args.representation_size,
                'mlp_dim': args.mlp_dim,
                'dropout': args.dropout,
            }
        elif net_name == 'nice':
            arch_params = {
                'encoder_emb': 40,
                'representation_size': args.representation_size
            }
        else:
            arch_params = {
                'channels_size': 63,
                'temporal_size': sfreq,
                'representation_size': args.representation_size
            }

    if net_name == 'transformer':
        eeg2clip = EEGTransformer(
            num_classes=1654 if args.concept_loss else 0,
            **arch_params
        )
    elif net_name == 'nice':
        eeg2clip = NiceNet(**arch_params)
    elif net_name == 'atm':
        eeg2clip = ATMNet(**arch_params)
    elif net_name == 'EEGNetv4':
        eeg2clip = EEGNetv4(**arch_params)
    else:
        sys.exit(f"{net_name} is not supported")

    if state_dict is not None:
        missing_keys, unexpected_keys = eeg2clip.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print("Missing keys", missing_keys)
        if len(unexpected_keys) > 0:
            print("Unexpected keys", unexpected_keys)
    return eeg2clip


def add_to_dict(src_dict, dst_dict, prefix=None):
    if prefix is None:
        prefix = ""
    for _key, _val in src_dict.items():
        if f"{prefix}{_key}" not in dst_dict:
            dst_dict[f"{prefix}{_key}"] = [_val]
        else:
            dst_dict[f"{prefix}{_key}"].append(_val)
    return dst_dict


def _add_common_arguments(parser):
    parser.add_argument("--eeg_dir", type=str, help="the directory with EEG data")
    parser.add_argument(
        "--feature_dir", default='../dnn_feature/', type=str,
        help="the directory for precomputed features"
    )
    parser.add_argument(
        "--feature_extractor", default=None, type=str,
        help="expects <feature_extractor>train and test.npy to exist in <feature_dir>"
    )
    parser.add_argument('--epochs', default=200, type=int, help="number of epochs")
    parser.add_argument('--subject', default=1, type=int, help='the participant number')
    parser.add_argument('--sfreq', default="250", type=str, help='the EEG frequency sample')
    parser.add_argument(
        "--single_trial", action=argparse.BooleanOptionalAction, default=False,
        help="if true EGG trials are not averaged",
    )
    parser.add_argument(
        "--avg_concepts", action=argparse.BooleanOptionalAction, default=False,
        help="if true EGG trials are averaged across concepts",
    )
    parser.add_argument("--clamp_th", type=int, default=None, help="outlier clamping threshold")
    parser.add_argument(
        "--val_trials", nargs='+', default=None, choices=[0, 1, 2, 3], type=int,
        help="the trial indices for validation"
    )
    parser.add_argument(
        "--val_imgs", nargs='+', default=None, choices=list(np.arange(10)), type=int,
        help="the indices of images for validation"
    )
    parser.add_argument(
        "--augmentations", nargs='+', default=None, type=str,
        help="training data augmentations"
    )

    parser.add_argument('-b', '--batch_size', default=256, type=int, help='dataloader batch size')
    parser.add_argument('-j', '--num_workers', default=0, type=int, help='dataloader workers')
    parser.add_argument('--lr', default=2e-4, type=float, help='the learning rate')
    parser.add_argument('--seed', default=2023, type=int, help='random seed')
    parser.add_argument("--out_dir", default='../results/', type=str, help="the output directory")
    parser.add_argument("--experiment_name", default='', type=str, help="the experiment name")
    parser.add_argument('--print_frequency', default=5, type=int,
                        help='the frequency of printing progress')

    parser.add_argument("--scheduler", nargs='+', default=None, type=str, help="the scheduler type")
    parser.add_argument("--optimiser", default='Adam', type=str, help="the optimiser type")

    parser.add_argument("--resume", default=None, type=str, help="the checkpoint to be resumed")
    parser.add_argument('--arch', default='transformer', type=str, help='networks type')
    parser.add_argument('--num_layers', default=8, type=int, help='transformer layers')
    parser.add_argument('--num_heads', default=8, type=int, help='transformer heads')
    parser.add_argument('--hidden_dim', default=128, type=int, help='transformer hidden dim')
    parser.add_argument('--dropout', default=0.5, type=float, help='transformer dropout')
    parser.add_argument('--representation_size', default=None, type=int,
                        help='transformer out size')
    parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp_dim')
    parser.add_argument(
        "--temporal_embedding", action=argparse.BooleanOptionalAction, default=False,
        help="if true the transformer embedding dimension is temporal axis",
    )

    parser.add_argument("--test", default=None, type=str, help="testing the checkpoint")
    parser.add_argument("--test_set", default=None, type=str, help="testing the checkpoint")
    return parser


def parse_arguments(parser):
    parser = _add_common_arguments(parser)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # whether to train-test within or acros subjects
    args.inter_subjects = True if args.subject < 0 else False
    args.subject = abs(args.subject)
    # whether testing or training
    if args.test is None:
        if args.feature_extractor is None:
            fdir_split = args.feature_dir.split('/')
            feature_dir = fdir_split[-2] if fdir_split[-1] == '' else fdir_split[-1]
            args.out_dir = f"{args.out_dir}/{feature_dir}__{args.arch}"
            if args.single_trial is False and args.avg_concepts is False:
                args.out_dir = f"{args.out_dir}__baseline"
            elif args.single_trial is False:
                args.out_dir = f"{args.out_dir}__avg-concept"
            else:
                args.out_dir = f"{args.out_dir}__resampling"
        args.out_dir = f"{args.out_dir}/sub-{args.subject:02d}{args.experiment_name}/"
    else:
        args.out_dir = f"{os.path.dirname(args.test)}/"
    # representation size
    if args.representation_size is None:
        args.representation_size = int(args.feature_dir.split('_')[-1].replace('/', ''))
    return args


def report_topks(features_from_eeg, features_from_img, label, predictions, gts, tops):
    similarity = features_from_eeg @ features_from_img.t()

    _, top5_indices = similarity.topk(5)
    predictions.extend(top5_indices.detach().cpu().tolist())
    gts.extend(label.detach().cpu().tolist())

    label = label.view(-1, 1)
    for top_i in tops.keys():
        top_pred = torch.any(label == top5_indices[:, :top_i], dim=1)
        tops[top_i].extend(top_pred.detach().cpu().tolist())
    return predictions, gts, tops


class Trainer:
    """The trainer class."""

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.eeg_dir = args.eeg_dir
        if args.feature_extractor is None:
            self.feature_extractor = f"{args.feature_dir}/"
        else:
            self.feature_extractor = f"{args.feature_dir}/{args.feature_extractor}_"
        self.subject = args.subject
        self.sfreq = args.sfreq
        self.val_imgs = args.val_imgs
        self.val_trials = args.val_trials
        self.out_dir = args.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.device = args.device
        self.print_frequency = args.print_frequency

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.epochs = args.epochs

        self.single_trial = args.single_trial
        self.avg_concepts = args.avg_concepts
        self.inter_subjects = args.inter_subjects
        self.temporal_embedding = args.temporal_embedding
        self.augmentations = args.augmentations
        self.clamp_th = args.clamp_th

        self.criterion_cls = nn.CrossEntropyLoss().to(args.device)

        self.arch = args.arch
        self.eeg2clip = load_model(args)
        self.eeg2clip = self.eeg2clip.to(args.device)

        if args.test is None:
            self.test_res_path = f"{self.out_dir}/test_output.csv"
            args.model_parameters = count_parameters(self.eeg2clip)
            print(f"Model with {args.model_parameters} parameters.")
            save_arguments(args)
        else:
            self.test_res_path = f"{self.out_dir}/{args.test_set}_{args.experiment_name}.csv"

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Optimizers
        if args.optimiser == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.eeg2clip.parameters(),
                ), lr=self.lr, betas=(0.9, 0.999)
            )
        else:
            self.optimizer = torch.optim.Adam(
                itertools.chain(
                    self.eeg2clip.parameters(),
                ), lr=self.lr, betas=(0.9, 0.999)
            )

        if args.scheduler is None:
            self.scheduler = None
        elif args.scheduler[0] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[int(m) for m in args.scheduler[1:]]
            )
        elif args.scheduler[0] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.epochs // 5,
                T_mult=1,
                eta_min=0
            )
        elif args.scheduler[0] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=float(args.scheduler[1]),
                patience=int(args.scheduler[2])
            )
        else:
            self.scheduler = None

        if args.resume is not None:
            self._load_resume_checkpoint(args.resume)

        print('Trainer is initialised!')

    def _load_resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.optimizer.load_state_dict(checkpoint['optimiser'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _save_network(self, filename):
        torch.save(
            {
                "state_dict": self.eeg2clip.state_dict(),
                "net_name": self.arch,
                "architecture": self.eeg2clip.save_params(),
                "optimiser": self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict()
            },
            f"{self.out_dir}/{filename}.pth"
        )

    def _save_losses(self, losses):
        df = pd.DataFrame(losses)
        df.to_csv(f"{self.out_dir}/losses.csv")

    def _get_eeg_data(self, train, avg=True):
        labels = np.arange(1654 if train else 200)
        file_name = f"preprocessed_eeg_{'training' if train else 'test'}.npy"
        # parsing the sfreq
        sfreq_parsed = self.sfreq.split('_')
        sfreq = int(sfreq_parsed[0])
        sfreq_suffix = f"_{sfreq_parsed[1]}" if len(sfreq_parsed) > 1 else ""
        data = np.load(
            f"{self.eeg_dir}/Preprocessed_data_{sfreq}Hz{sfreq_suffix}/sub-{self.subject:02d}/{file_name}",
            allow_pickle=True
        )
        data = data['preprocessed_eeg_data']
        if avg:
            data = np.mean(data, axis=1)
        return data, labels

    def _get_features(self, split):
        features = np.load(f"{self.feature_extractor}{split}.npy", allow_pickle=True)
        return features

    def generic_test(self, save_print=False):
        self.eeg2clip.eval()

        eeg_signal, labels = self._get_eeg_data(train=False, avg=True)
        eeg_signal = torch.from_numpy(eeg_signal).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.LongTensor)
        dataset = torch_data.TensorDataset(eeg_signal, labels)
        db_loader = torch_data.DataLoader(
            dataset, self.batch_size, num_workers=self.num_workers, shuffle=False
        )

        features_from_img = torch.from_numpy(self._get_features('test_out_distribution')).type(
            torch.FloatTensor)
        features_from_img = features_from_img / features_from_img.norm(dim=1, keepdim=True)
        features_from_img = features_from_img.to(self.device)

        losses = {'total': []}
        predictions = []
        gts = []
        tops = {1: [], 3: [], 5: []}
        with torch.no_grad():
            for i, (eeg_signal, label) in enumerate(db_loader):
                data_len = len(eeg_signal)
                eeg_signal = eeg_signal.to(self.device)
                label = label.to(self.device)
                features_target = torch.arange(data_len).to(self.device)

                _, features_from_eeg = self.eeg2clip(eeg_signal)
                features_from_eeg = features_from_eeg / features_from_eeg.norm(dim=1, keepdim=True)
                # cosine similarity as the logits
                logits_eeg = self.logit_scale.exp() * features_from_eeg @ features_from_img.t()
                logits_img = logits_eeg.t()
                loss_eeg_cosine = self.criterion_cls(logits_eeg, features_target)
                loss_img_cosine = self.criterion_cls(logits_img, features_target)
                loss = loss_eeg_cosine * 0.5 + loss_img_cosine * 0.5
                losses['total'].extend([loss.item()] * data_len)

                predictions, gts, tops = report_topks(
                    features_from_eeg, features_from_img, label, predictions, gts, tops
                )

        losses = {key: np.mean(val) for key, val in losses.items()}
        test_results = {'predictions': predictions, 'gts': gts, 'tops': tops}
        if save_print:
            self._save_test_predictions(test_results)
        tops = {f"top{key}": np.mean(val) for key, val in tops.items()}
        return {'loss': losses['total']}, tops

    def _save_test_predictions(self, tests):
        predictions = tests['predictions']
        gts = tests['gts']
        tops = tests['tops']

        df = pd.DataFrame(predictions, columns=[f"pred{i}" for i in range(5)])
        df.insert(len(df.columns), 'gt', gts)
        for top_i, top_i_pred in tops.items():
            df.insert(len(df.columns), f"top{top_i}", top_i_pred)
        df.to_csv(self.test_res_path)
        print(f"The test Top1 [{np.mean(tops[1]):.02f}] Top5 [{np.mean(tops[5]):.02f}]")
