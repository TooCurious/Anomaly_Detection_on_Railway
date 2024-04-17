import cv2
import numpy as np
import os
from tqdm import tqdm


def line_search(path_file: str):
    """Находим линию разделяющую белую и черную области маски
    Args:
        path_file (str): путь до файла
    """
    # Загрузка бинарного изображения
    binary_image = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

    # Нахождение границ объекта с помощью оператора Canny
    edges = cv2.Canny(binary_image, 50, 150)

    # Применение преобразования Хафа для обнаружения прямых линий
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    x1_0 = []
    y1_0 = []
    x2_0 = []
    y2_0 = []
    # Если есть обнаруженные линии
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Преобразование параметров линии в координаты
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Округление координат и перевод в целые числа
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            x1_0.append(x1)
            y1_0.append(y1)
            x2_0.append(x2)
            y2_0.append(y2)

    return x1_0[0], y1_0[0], x2_0[0], y2_0[0]


def reflect_image(image, line_start, line_end):
    # Находим уравнение прямой, проходящей через две точки
    x1, y1 = line_start
    x2, y2 = line_end

    # Находим координаты вектора, параллельного прямой
    dx = x2 - x1
    dy = y2 - y1

    # Находим координаты вектора, перпендикулярного прямой
    perpendicular_dx = -dy
    perpendicular_dy = dx

    # Находим расстояние от точек до прямой
    distances = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Находим вектор от точки до начальной точки прямой
            vector_x = x - x1
            vector_y = y - y1
            # Находим скалярное произведение вектора от точки до начальной точки прямой
            # и вектора, перпендикулярного прямой
            scalar_product = vector_x * perpendicular_dx + vector_y * perpendicular_dy
            # Нормируем скалярное произведение для получения расстояния от точки до прямой
            distance = scalar_product / np.sqrt(perpendicular_dx ** 2 + perpendicular_dy ** 2)
            distances.append(distance)

    # Находим максимальное расстояние (это будет расстоянием от прямой до крайней точки)
    distance = max(distances)

    # Создаем изображение для отражения
    reflected_image = np.zeros_like(image)

    # Отражаем пиксели относительно заданной линии
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Находим вектор от точки до начальной точки прямой
            vector_x = x - x1
            vector_y = y - y1
            # Находим скалярное произведение вектора и вектора, перпендикулярного прямой
            scalar_product = vector_x * perpendicular_dx + vector_y * perpendicular_dy
            # Находим точку, отраженную относительно прямой
            reflected_x = x - 2 * scalar_product * perpendicular_dx / (perpendicular_dx ** 2 + perpendicular_dy ** 2)
            reflected_y = y - 2 * scalar_product * perpendicular_dy / (perpendicular_dx ** 2 + perpendicular_dy ** 2)
            # Проверяем, чтобы новые координаты пикселя были в пределах изображения
            if 0 <= int(reflected_y) < image.shape[0] and 0 <= int(reflected_x) < image.shape[1]:
                reflected_image[y, x] = image[int(reflected_y), int(reflected_x)]

    return reflected_image


def masks(mask, image, invert=False):
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if invert:
        binary_mask = cv2.bitwise_not(binary_mask)
    cropped_image = cv2.bitwise_and(image, image, mask=binary_mask)

    return cropped_image


def train_image(path_1, path_2, path_3):
    # смотри colab "transformation_image.ipynb"
    mask_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
    mask_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(path_3)

    x1, y1, x2, y2 = line_search(path_1)
    line_start = (x1, y1)
    line_end = (x2, y2)

    # Отражаем изображение относительно заданной линии
    reflected_image = reflect_image(image, line_start, line_end)

    cut_image = masks(mask_1, image)
    cut_reflected_image = masks(mask_1, reflected_image, invert=True)

    stitched_image = cv2.add(cut_image, cut_reflected_image)

    x1, y1, x2, y2 = line_search(path_2)
    line_start = (x1, y1)
    line_end = (x2, y2)

    # Отражаем изображение относительно заданной линии
    reflected_image = reflect_image(stitched_image, line_start, line_end)

    cut_image = masks(mask_2, stitched_image)
    cut_reflected_image = masks(mask_2, reflected_image, invert=True)

    stitched_image = cv2.add(cut_image, cut_reflected_image)

    return stitched_image


# input_folder = 'C:/Users/User/Desktop/category_2/test_with_obstacle/anomaly'
# output_folder = 'C:/Users/User/Desktop/category_2/test_preparing/anomaly'
#
# # Проверка существования папки вывода и создание ее, если она не существует
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Проход по всем файлам в папке ввода
# for filename in tqdm(os.listdir(input_folder)):
#     # Проверка, является ли файл изображением
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # Чтение изображения
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path)
#
#         # Применение функции к изображению
#         path_1 = 'C:/Users/User/Desktop/category_3/mask.png'
#         # path_2 = 'C:/Users/User/Desktop/category_4/mask_2.png'
#
#         processed_image = train_image(path_1, path_2, os.path.join(input_folder, filename))
#
#         # Сохранение обработанного изображения в папку вывода
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, processed_image)


# # Проход по всем файлам в папке ввода
# for filename in tqdm(os.listdir(input_folder)):
#     # Проверка, является ли файл изображением
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # Чтение изображения
#         mask_1 = cv2.imread('C:/Users/User/Desktop/category_2/mask.png', cv2.IMREAD_GRAYSCALE)
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path)
#
#         processed_image = masks(mask_1, image)
#
#         # Сохранение обработанного изображения в папку вывода
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, processed_image)


#
# image = cv2.imread('C:/Users/User/Desktop/category_2/test_with_obstacle/anomaly/173_11.png')
# print(type(image))