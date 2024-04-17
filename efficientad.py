#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from metrics import compute_anomaly_detection_metrics
from preprocessing import masks as auxiliary_function
import json

path_to_dataset = 'C:/Users/User/Desktop/category_3'
path_to_ImageNet = 'C:/Users/User/Downloads/ILSVRC/Data/CLS-LOC/train'
path_to_truth_mask_segmentation = 'C:/Users/User/Desktop/category_3/SegmentationClass'
path_to_mask_camera = 'C:/Users/User/Desktop/category_3/mask.png'
path_to_model = 'output/pretraining/1/teacher_small_final_state.pth'


# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

grayscale_transform = transforms.RandomGrayscale(p=1.0)

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])


def random_brightness_region(image_tensor, brightness_factor_range=(0.2, 3.0), region_size=(100, 100), N=10):
    """
    Randomly adjusts the brightness of a randomly selected region in the image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor (C x H x W).
        brightness_factor_range (tuple): Range for random brightness factor.
        region_size (tuple): Size of the randomly selected region (H x W).

    Returns:
        torch.Tensor: Image tensor with brightness adjusted in the randomly selected region.
    """
    # Ensure input tensor is in range [0, 1]
    if image_tensor.min() < 0 or image_tensor.max() > 1:
        raise ValueError("Input image tensor must be in the range [0, 1]")

    adjusted_image_tensor = image_tensor.clone()

    for _ in range(N):
      # Get image size
      _, height, width = image_tensor.size()

      # Randomly select top-left corner of region
      top = random.randint(0, height - region_size[0])
      left = random.randint(0, width - region_size[1])

      # Randomly adjust brightness factor
      brightness_factor = random.uniform(brightness_factor_range[0], brightness_factor_range[1])

      # Apply brightness adjustment to the selected region
      region = image_tensor[:, top:top+region_size[0], left:left+region_size[1]]
      adjusted_region = TF.adjust_brightness(region, brightness_factor)

      # Paste adjusted region back into original image tensor
      adjusted_image_tensor[:, top:top+region_size[0], left:left+region_size[1]] = adjusted_region

    return adjusted_image_tensor


def train_transform(image):
    image = grayscale_transform(image)
    #image = random_brightness_region(image, brightness_factor_range=(0.2, 1.0), region_size=(200, 200))
    return default_transform(transform_ae(image)), default_transform(transform_ae(image))


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_path = path_to_dataset

    pretrain_penalty = True

    # create output dir
    train_output_dir = os.path.join('output/category_3', 'trainings')
    test_output_dir = os.path.join('output/category_3', 'anomaly_maps', 'test')
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, 'full_train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, 'test_preparing'))

    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(1),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
        ])
        penalty_set = ImageFolderWithoutTarget(path_to_ImageNet,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)
    state_dict = torch.load(path_to_model, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    #teacher_mean, teacher_std = torch.load('teacher_stats.pt')
    torch.save((teacher_mean, teacher_std), 'teacher_stats.pt')


    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * 170000), gamma=0.1)
    tqdm_obj = tqdm(range(170000))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty ** 2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()


        result = {}
        result['train_loss'] = []
        # result['pro_auc'] = []
        # result['roc_auc'] = []
        # result['iou_auc'] = []

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))
            result['train_loss'].append(loss_total.item())

        if iteration % 10000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))

        # if iteration % 100 == 0 and iteration > 0:
        #     # run intermediate evaluation
        #     teacher.eval()
        #     student.eval()
        #     autoencoder.eval()
        #
        #
        #     # q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        #     #     validation_loader=validation_loader, teacher=teacher,
        #     #     student=student, autoencoder=autoencoder,
        #     #     teacher_mean=teacher_mean, teacher_std=teacher_std,
        #     #     desc='Intermediate map normalization')
        #     # pro_auc, roc_auc, iou_auc= test(
        #     #     test_set=test_set, teacher=teacher, student=student,
        #     #     autoencoder=autoencoder, teacher_mean=teacher_mean,
        #     #     teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        #     #     q_ae_start=q_ae_start, q_ae_end=q_ae_end, desc='Intermediate inference')
        #     # print('Intermediate image pro_auc: {:.4f}'.format(pro_auc))
        #     # print('Intermediate image roc_auc: {:.4f}'.format(roc_auc))
        #     # print('Intermediate image iou_auc: {:.4f}'.format(iou_auc))
        #     # print('Intermediate image detect precision: {:.4f}'.format(precision_value))
        #     # print('Intermediate image detect recall: {:.4f}'.format(recall_value))
        #     # print('Intermediate image F-metric: {:.4f}'.format(f_value))
        #     # сохраняем значения в словарь
        #     # result['pro_auc'].append(pro_auc)
        #     # result['roc_auc'].append(roc_auc)
        #     # result['iou_auc'].append(iou_auc)
        #     # teacher frozen
        #     teacher.eval()
        #     student.train()
        #     autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher.state_dict(), os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student.state_dict(), os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder.state_dict(), os.path.join(train_output_dir, 'autoencoder_final.pth'))

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    pro_auc, roc_auc, iou_auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end, desc='Final inference')
    print('Final image pro_auc: {:.4f}'.format(pro_auc))
    print('Final image roc_auc: {:.4f}'.format(roc_auc))
    print('Final image iou_auc: {:.4f}'.format(iou_auc))
    # print('Final image detect precision: {:.4f}'.format(precision_value))
    # print('Final image detect recall: {:.4f}'.format(recall_value))
    # print('Final image F-metric: {:.4f}'.format(f_value))
    # result['pro_auc'].append(pro_auc)
    # result['roc_auc'].append(roc_auc)
    # result['iou_auc'].append(iou_auc)

    with open('my_dict.json', 'w') as json_file:
        json.dump(result, json_file)


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, desc='Running inference'):

    amaps = []
    masks = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = grayscale_transform(image)
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        mask_1 = cv2.imread(path_to_mask_camera, cv2.IMREAD_GRAYSCALE)
        map_combined = auxiliary_function(mask_1, map_combined)
        amaps.append(map_combined)

        split_path = path.split('\\')
        desired_path = '\\'.join(split_path[2:])
        gray_image = cv2.imread(os.path.join(path_to_truth_mask_segmentation, desired_path),
                                cv2.IMREAD_GRAYSCALE) / 255
        masks.append(gray_image)

    masks = np.array(masks)
    amaps = np.array(amaps)

    pro_auc, roc_auc, iou_auc = compute_anomaly_detection_metrics(masks, amaps)

    return pro_auc, roc_auc, iou_auc



@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:]) ** 2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


if __name__ == '__main__':
    main()
