"""
Extracting features from pretrained networks.
"""

import os
import numpy as np
import glob
import argparse

from PIL import Image as pil_image
import torch
import torchvision.transforms as torch_transforms

import osculari
import open_clip
import clip


def read_img(img_name, things_imgs_dir):
    concept = img_name[:-8]
    img_path = f"{things_imgs_dir}{concept}/{img_name}"
    return pil_image.open(img_path).convert("RGB")


def test_out_distribution_set(model, preprocess, extractor_fun, feature_name, dnn_feature_dir,
                              things_imgs_dir, out_dist_dir, language_dir=None):
    centre_imgs = sorted(glob.glob(f"{out_dist_dir}/*/*.jpg"))
    centre_imgs = [img_path.split('/')[-1] for img_path in centre_imgs]

    sind = None
    current_concept = None
    concept_inds = []
    for img_ind, img_name in enumerate(centre_imgs):
        concept = img_name[:-8]
        if sind is None:
            sind = img_ind
            current_concept = concept
        elif concept != current_concept or img_ind == len(centre_imgs) - 1:
            eind = img_ind
            concept_inds.append([sind, eind])
            sind = eind
            current_concept = concept

    if language_dir is None:
        db_elements = centre_imgs
    else:
        db_elements = f"{language_dir}/blip_texts_things_egg2_control.npy"

    features = extractor_fun(db_elements, model, preprocess, things_imgs_dir)
    features = np.array(features)

    feature_size = np.array(features).shape[1]

    dnn_feature_dir = f"{dnn_feature_dir}/{feature_name}__{feature_size}"
    os.makedirs(dnn_feature_dir, exist_ok=True)

    np.save(f"{dnn_feature_dir}/things_egg2_control.npy", features)

    avg = []
    for cind in concept_inds:
        avg.append(np.mean(features[cind[0]:cind[1]], axis=0))
    return np.array(avg), dnn_feature_dir


def make_datasets(model, preprocess, extractor_fun, feature_name, dnn_feature_dir, eeg_metadata,
                  things_imgs_dir, out_dist_dir, language_dir=None):
    # out of distrbution
    print('Doing test out distribution.')
    features, dnn_feature_dir = test_out_distribution_set(
        model, preprocess, extractor_fun, feature_name, dnn_feature_dir, things_imgs_dir,
        out_dist_dir, language_dir
    )

    np.save(f"{dnn_feature_dir}/test_out_distribution.npy", features)
    print(np.array(features).shape)

    # in distribution
    print('Doing test in distribution.')
    if language_dir is not None:
        db_elements = f"{language_dir}/blip_texts_test_in_distribution.npy"
    else:
        db_elements = eeg_metadata['test_img_files']
    features = extractor_fun(db_elements, model, preprocess, things_imgs_dir)
    np.save(f"{dnn_feature_dir}/test_in_distribution.npy", features)
    print(np.array(features).shape)

    # train
    print('Doing training.')
    if language_dir is not None:
        db_elements = f"{language_dir}/blip_texts_train.npy"
    else:
        db_elements = eeg_metadata['train_img_files']
    features = extractor_fun(db_elements, model, preprocess, things_imgs_dir)
    np.save(f"{dnn_feature_dir}/train.npy", features)
    print(np.array(features).shape)


def extract_visual_features(image_set, model, preprocess, things_imgs_dir, batch_size=64):
    visual = []
    imgs = []
    for img_ind, img_name in enumerate(image_set):
        image = read_img(img_name, things_imgs_dir)
        imgs.append(image)
        if len(imgs) == batch_size or img_ind == len(image_set) - 1:
            input_imgs = [preprocess(img) for img in imgs]
            input_imgs = torch.tensor(np.stack(input_imgs)).cuda()
            with torch.no_grad():
                image_features = model(input_imgs).cpu().numpy()
                if len(image_features.shape) == 3:
                    image_features = image_features[:, 0]
                elif len(image_features.shape) == 5:
                    image_features = image_features.transpose(0, 2, 1, 3, 4)[:, :, 0, 0, 0]
                # image_features = np.concatenate(
                #     [image_features[:, 0], image_features[:, 1:].mean(axis=1)], axis=1
                # )
                visual.extend(image_features)
            imgs = []
    return visual


def extract_language_features(text_file, model, _preprocess, _things_imgs_dir, batch_size=64):
    text_set = np.load(text_file, allow_pickle=True)
    language = []
    texts = []
    for text_ind, text in enumerate(text_set):
        texts.append(text)
        if len(texts) == batch_size or text_ind == len(text_set) - 1:
            input_texts = clip.tokenize(texts).cuda()
            with torch.no_grad():
                text_features = model.encode_text(input_texts).cpu().numpy()
                language.extend(text_features)
            texts = []
    return language


def main():
    parser = argparse.ArgumentParser(description='TExtracting features from DNNs.')
    parser.add_argument("--out_dir", type=str, help="the directory with EEG data")
    parser.add_argument("--things_dir", type=str, help="the things dataset")
    parser.add_argument("--out_dist_dir", type=str, help="the test set images")

    parser.add_argument("--architecture", type=str, help="the network architecture")
    parser.add_argument("--layer", type=str, help="the layer")
    parser.add_argument("--pooling", action="store_true", default=False)
    parser.add_argument("--language_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    if 'openclip_' in args.architecture:
        architecture = args.architecture.replace('openclip_', '')
        layer = 'visual' if args.language_dir is None else 'language'
        pretrained_weights = open_clip.list_pretrained_tags_by_model(architecture)
        pretrained = pretrained_weights[int(args.layer)]
        print(pretrained, pretrained_weights)
        feature_extractor, _, transforsm = open_clip.create_model_and_transforms(
            architecture,
            pretrained=pretrained,
            precision='fp32',
            device='cuda'
        )
        if args.language_dir is None:
            feature_extractor = feature_extractor.encode_image
        feature_name = f"{architecture}__{pretrained}__{layer}"
    else:
        architecture = args.architecture  # network's architecture
        weights = architecture  # the pretrained weights
        layer = args.layer  # the readout layer

        readout_kwargs = {  # parameters for extracting features from the pretrained network
            'architecture': architecture,
            'weights': weights,
            'layers': layer,
            'pooling': 'avg_1_1' if args.pooling else None
        }
        feature_extractor = osculari.models.FeatureExtractor(**readout_kwargs)
        feature_extractor.cuda()

        img_size = args.img_size
        mean, std = feature_extractor.normalise_mean_std
        # converting it to torch tensor
        transforsm = torch_transforms.Compose([
            torch_transforms.Resize((img_size, img_size)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=mean, std=std)
        ])
        feature_name = f"{architecture}__{layer}"

    things_eeg_dir = f"{args.things_dir}/EEG2/"
    eeg_metadata = np.load(f"{things_eeg_dir}image_metadata.npy", allow_pickle=True)[()]

    things_imgs_dir = f"{args.things_dir}THINGS/Images/"

    if args.language_dir is None:
        extract_fun = extract_visual_features
    else:
        extract_fun = extract_language_features
    make_datasets(
        feature_extractor, transforsm, extract_fun, feature_name, args.out_dir, eeg_metadata,
        things_imgs_dir, args.out_dist_dir, args.language_dir
    )


if __name__ == "__main__":
    main()
