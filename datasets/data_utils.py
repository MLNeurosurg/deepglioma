import random
import time
from collections import Counter, OrderedDict, defaultdict
from tifffile import imread
import os
from typing import List, Counter
import torch
from torchvision import transforms
import random
import pandas
import numpy as np
from typing import List, Tuple
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_labels(df, label_list):
    cols = ['dbcase', 'center', 'class']
    cols.extend(label_list)
    df = df.loc[:, cols]
    return df


def get_glioma_data(study_df: pandas.DataFrame, labels_df: pandas.DataFrame,
                    data_root_path: str) -> List:

    # iterate over the labelled cases
    data = []
    for i in range(labels_df.shape[0]):

        # get the case and labels
        db_case = labels_df.iloc[i, :].dbcase
        labels = labels_df.iloc[i, 3:].tolist()
        series_df = study_df[study_df.study.str.contains(db_case)]

        # skip if series should not be included
        if series_df.empty:
            print('{} not included as study case.'.format(db_case))
            continue

        else:
            for series in series_df.series:
                try:
                    patch_path = os.path.join(labels_df.iloc[i, :].center,
                                              labels_df.iloc[i, :].dbcase,
                                              str(int(series)), 'data',
                                              'patches', 'tumor')
                except ValueError:
                    patch_path = os.path.join(labels_df.iloc[i, :].center,
                                              labels_df.iloc[i, :].dbcase,
                                              str(series), 'data', 'patches',
                                              'tumor')

                try:
                    # read in files the series
                    files = os.listdir(os.path.join(data_root_path,
                                                    patch_path))
                    for file in files:
                        # get file path
                        file_path = os.path.join(patch_path, file)
                        instance_dict = {}
                        instance_dict['objects'] = labels
                        instance_dict['file_name'] = file_path
                        data.append(instance_dict)
                except FileNotFoundError:
                    print('{} not found in database'.format(patch_path))
                    continue
    return data


def get_ssl_data(study_df: pandas.DataFrame, data_root_path: str) -> List:
    data = []
    for file in study_df.file_name:
        instance_dict = {}
        instance_dict['objects'] = [1]
        instance_dict['file_name'] = file[1:]
        data.append(instance_dict)
    return data


def get_all_normal_diagnostic(root_path):
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if ('normal' in root) or ('tumor' in root):
                file_paths.append(os.path.join(root, file))
    return sorted(file_paths)


def check_data(data: list, level='slide'):
    cases = defaultdict(int)
    for patch in data:
        patch_path = patch['file_name']
        slide = patch_path.partition('data')[0]
        cases[slide] += 1
    return cases


def convert_unk_labels(df: pandas.DataFrame, label: str):

    for idx, i in enumerate(df[label]):
        try:
            if np.isnan(i):
                df[label][idx] = 'UNK'
        except:
            continue
    assert df[label].unique().shape[0] == 3, 'Labels not correctly changed.'
    return df


def train_validation_split(data: List, validation_cases: List) -> Tuple:
    """Function to split the data into training and validation cases
	based on validation_cases list."""
    val_data = []
    train_data = []
    for i in data:
        for val_case in validation_cases:
            if val_case in i['file_name']:
                val_data.append(i)
                break
        else:
            train_data.append(i)

    assert len(train_data) + len(val_data) == len(data)
    for val in validation_cases:
        for i in train_data:
            if val in i['file_name']:
                print("WARNING: VALIDATION CASES in TRAINING DATA!!!")
    return train_data, val_data


def oversample_label(data: List,
                     label_index: int,
                     perc_majority_label: float = 1) -> List:
    """Function to better balance the data. perc_majority_label is what percentage
	you wish to reach of the majority class. A value of 1 means the function will
	oversample until the classes are balanced."""

    neg_samples = []
    pos_samples = []

    # iterate over data
    for sample in data:
        if sample['objects'][label_index] == 1:
            pos_samples.append(sample)
        else:
            neg_samples.append(sample)

    num_neg = len(neg_samples)
    num_pos = len(pos_samples)
    assert num_neg > num_pos, 'Label is already majority of samples.'

    num_needed = int((num_neg - num_pos) * perc_majority_label)
    oversampled_samples = np.random.choice(pos_samples,
                                           size=num_needed,
                                           replace=True)
    neg_samples.extend(oversampled_samples)
    neg_samples.extend(pos_samples)

    return neg_samples


def label_frequency(data: List, n_labels: int = 3) -> Tuple:
    """Function that will count the overall label frequency in a dataset."""
    label_counter = Counter()
    count = 0
    l = np.zeros(shape=(len(data), n_labels))
    for sample in data:
        label_counter.update(
            [i for i, x in enumerate(sample['objects']) if x != 0])
        l[count, :] = sample['objects']
        count += 1
    for key in label_counter:
        label_counter[key] /= count

    return label_counter, l


def tsne_data(data: List,
              label_index: int,
              n_samples_label: int = 1000) -> Tuple:

    neg_samples = []
    pos_samples = []

    # iterate over data
    for sample in data:
        if sample['objects'][label_index] == 1:
            pos_samples.append(sample)
        else:
            neg_samples.append(sample)

    pos_samples = np.random.choice(pos_samples,
                                   size=n_samples_label,
                                   replace=n_samples_label > len(pos_samples))

    neg_samples = np.random.choice(neg_samples,
                                   size=n_samples_label,
                                   replace=n_samples_label > len(neg_samples))

    return pos_samples, neg_samples


def image_transforms(image_size=300, strength='weak'):
    """Get strong or weak image tranformations."""
    if strength == 'strong':
        print('You are using STRONG data augmentations.')
        srh_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=image_size,
                                             scale=(0.3, 1),
                                             ratio=(1, 1),
                                             interpolation=2)
            ],
                                   p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
            transforms.RandomAdjustSharpness(3, p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(p=0.3)
        ])
    else:
        srh_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=image_size, interpolation=2)
        ])

    return srh_transforms


def get_unk_mask_indices(image, testing, num_labels, known_labels):
    if testing:
        # random seed with image for consistent results
        np.random.seed(image.astype(np.uint8))
        unk_mask_indices = random.sample(range(num_labels),
                                         (num_labels - int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels > 0:
            # assumes that at minumum, 25% of the labels will be present
            # num_known = random.randint(0, int(num_labels * 0.75))
            num_known = known_labels
        else:
            num_known = 0
        unk_mask_indices = random.sample(range(num_labels),
                                         (num_labels - num_known))
    return unk_mask_indices


def get_third_channel(two_channel_image):
    """Helper function to generate our third channel from our two channel images."""
    img = np.zeros((two_channel_image.shape[0], two_channel_image.shape[1], 3),
                   dtype=float)

    CH2 = two_channel_image[:, :, 0].astype(float)
    CH3 = two_channel_image[:, :, 1].astype(float)
    subtracted_channel = (CH3 - CH2) + 5000
    subtracted_channel[subtracted_channel < 0] = 0

    img[:, :, 0] = subtracted_channel
    img[:, :, 1] = CH2
    img[:, :, 2] = CH3
    return img


def image_loader(path, transform=None):
    try:
        image = imread(path)
    # handle IO errors during training
    except FileNotFoundError:
        time.sleep(10)
        image = imread(path)

    image = np.moveaxis(image, 0, -1)
    image = get_third_channel(image)
    # rescale images between 0 and 1
    image /= image.max()

    if transform is not None:
        image = transform(image)

    return image
