#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from anomalib.utils.metrics import AUPRO
import cv2
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from metrics import harmonic_mean, insert_over_union, PL, compute_anomaly_detection_metrics

from efficientad import teacher_normalization, map_normalization, predict
from preprocessing import masks as auxiliary_function

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


def valid_transform(image):
    image = grayscale_transform(image)
    return default_transform(image), default_transform(image)


def inference():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_path = 'C:/Users/User/Desktop/category_4'


    # create output dir
    test_output_dir = os.path.join('inference/2', 'anomaly_maps', 'test')
    test_bound_output_dir = os.path.join('inference/2', 'bounding', 'test')
    os.makedirs(test_bound_output_dir)
    os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, 'full_train'),
        transform=transforms.Lambda(valid_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, 'test_preparing'))

    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1)

    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)
    state_dict_teacher = torch.load('output/2/trainings/teacher_final.pth', map_location='cpu')
    state_dict_student = torch.load('output/2/trainings/student_final.pth', map_location='cpu')
    teacher.load_state_dict(state_dict_teacher)
    student.load_state_dict(state_dict_student)
    state_dict_autoencoder = torch.load('output/2/trainings/autoencoder_final.pth', map_location='cpu')
    autoencoder = get_autoencoder(out_channels)
    autoencoder.load_state_dict(state_dict_autoencoder)

    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = torch.load('teacher_stats.pt')

    teacher.eval()
    student.eval()
    autoencoder.eval()

    # q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
    #     validation_loader=validation_loader, teacher=teacher, student=student,
    #     autoencoder=autoencoder, teacher_mean=teacher_mean,
    #     teacher_std=teacher_std, desc='Final map normalization')
    # torch.save((q_st_start, q_st_end, q_ae_start, q_ae_end), 'final_map_normalization.pt')
    q_st_start, q_st_end, q_ae_start, q_ae_end = torch.load('final_map_normalization.pt')
    # pro_auc, roc_auc, iou_auc, precision_value, recall_value, f_value, optim_treshold = test(
    #     test_set=test_set, teacher=teacher, student=student,
    #     autoencoder=autoencoder, teacher_mean=teacher_mean,
    #     teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
    #     q_ae_start=q_ae_start, q_ae_end=q_ae_end,
    #     test_output_dir=test_output_dir, test_bound_output_dir=test_bound_output_dir, treshold=0.2, desc='Final inference')
    print(test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, test_bound_output_dir=test_bound_output_dir, treshold=0.25, desc='Final inference'))
    # print('Final image pro_auc: {:.4f}'.format(pro_auc))
    # print('Final image roc_auc: {:.4f}'.format(roc_auc))
    # print('Final image iou_auc: {:.4f}'.format(iou_auc))
    # print('Final image detect precision: {:.4f}'.format(precision_value))
    # print('Final image detect recall: {:.4f}'.format(recall_value))
    # print('Final image F-metric: {:.4f}'.format(f_value))
    # print('Final image treshold: {:.4f}'.format(optim_treshold))


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None, test_bound_output_dir=None, treshold=None,
         desc='Running inference'):

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
        map_combined = torch.nn.functional.pad(map_combined, (2, 2, 2, 2))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bicubic')
        map_combined = map_combined[0, 0].cpu().numpy()
        mask_1 = cv2.imread('C:/Users/User/Desktop/category_4/mask.png', cv2.IMREAD_GRAYSCALE)
        map_combined = auxiliary_function(mask_1, map_combined)
        amaps.append(map_combined)
        map_combined_for_write = np.where(map_combined > treshold, 255, 0)

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.png')
            cv2.imwrite(file, map_combined_for_write)

        cv_array = map_combined_for_write.astype(np.uint8)
        _, thresh = cv2.threshold(cv_array, 254, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        split_path = path.split('\\')
        desired_path = '\\'.join(split_path[2:])
        image = cv2.imread(os.path.join('C:/Users/User/Desktop/category_4/test', desired_path))
        gray_image = cv2.imread(os.path.join('C:/Users/User/Desktop/category_4/SegmentationClass', desired_path),
                                cv2.IMREAD_GRAYSCALE) / 255
        masks.append(gray_image)

        contour_properties = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            # подгоняем рамки
            # x = x * 1280 / 256
            # y = y * 720 / 256
            # width = width * 1280 / 256
            # height = height * 720 / 256
            Xmax = x + width
            Xmin = x
            Ymax = y + height
            Ymin = y
            contour_properties.append((Xmax, Xmin, Ymax, Ymin))

            cv2.rectangle(image, (Xmin, Ymin), (Xmax, Ymax), (0, 0, 255), 2)

        if test_bound_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_bound_output_dir, defect_class)):
                os.makedirs(os.path.join(test_bound_output_dir, defect_class))
            file = os.path.join(test_bound_output_dir, defect_class, img_nm + '.png')
            cv2.imwrite(file, image)

    amaps = np.array(amaps)
    masks = np.array(masks)

    # pro_auc, roc_auc, iou_auc, precision_value, recall_value, f_value, optim_treshold = compute_anomaly_detection_metrics(masks, amaps)
    #
    # return pro_auc, roc_auc, iou_auc, precision_value, recall_value, f_value, optim_treshold




if __name__ == '__main__':
    inference()
