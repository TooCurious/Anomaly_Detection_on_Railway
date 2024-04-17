import cv2
import numpy as np
import os
from tqdm import tqdm

# Путь к папке с изображениями
folder_path = 'C:/Users/User/Desktop/category_3/train_preparing/anomaly'
output_folder = 'C:/Users/User/Desktop/category_3/full_train/anomaly'

# Проверка существования папки вывода и создание ее, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Получить список файлов из папки
image_files = os.listdir(folder_path)

# Значения alpha
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Проход по всем файлам изображений в папке
for i, image_file_1 in tqdm(enumerate(image_files[:57])):
    for image_file_2 in image_files[i+1:]:
        # Загрузка изображений
        image1 = cv2.imread(os.path.join(folder_path, image_file_1))
        image2 = cv2.imread(os.path.join(folder_path, image_file_2))

        # Применение blend для каждого значения alpha
        for alpha in alphas:
            blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

            # Сохранение результата
            output_file = f"blend_{alpha}_{image_file_1}_{image_file_2}"

            output_path = os.path.join(output_folder, output_file)
            cv2.imwrite(output_path, blended_image)
