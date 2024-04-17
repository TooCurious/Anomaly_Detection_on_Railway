from PIL import Image, TiffImagePlugin
import tifffile
import numpy as np
import cv2

# Пример использования функции find_rectangles
image_path = "C:/Users/User/Desktop/RZD/pythonProject/output/1/anomaly_maps/test/anomaly/1837_7.tiff"

# Считывание изображения из файла TIFF
image = tifffile.imread(image_path)
# Преобразование изображения в массив numpy
image_array = np.asarray(image)

# Вывод массива numpy
# max_value = np.amax(image_array)
# min_value = np.amin(image_array)
#
# condition = image_array > 0.5
# count = np.sum(condition)
# count_1 = np.size(image_array)
#
# # Вывод максимального значения
# print("Максимальное значение:", max_value)
# print("Минимальное значение:", min_value)
# print("count:", count)
# print("count_1:", count_1)

arr = np.where(image_array > 0.1, 255, 0)

# max_value = np.amax(arr)
# min_value = np.amin(arr)
# condition = arr > 1
# count = np.sum(condition)
# count_1 = np.size(arr)
#
# # Вывод максимального значения
# print("Максимальное значение:", max_value)
# print("Минимальное значение:", min_value)
# print("count:", count)
# print("count_1:", count_1)

tifffile.imwrite("C:/Users/User/Desktop/66_11_copy.tiff", arr)

# Открываем изображение TIFF
image = Image.open("C:/Users/User/Desktop/66_11_copy.tiff")

# Отображаем изображение
image.show()

gray_image = image.convert('L')

gray_image.save('C:/Users/User/Desktop/66_11_copy.png', 'PNG')

# Загрузите изображение
image_main = cv2.imread("C:/Users/User/Desktop/1837_7.png")
image = cv2.imread("C:/Users/User/Desktop/66_11_copy.png")

# Преобразуйте изображение в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Примените пороговую обработку для получения черно-белого изображения
ret, thresh = cv2.threshold(gray, 254, 255, 0)

# Найдите контуры в изображении
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Нарисуйте контуры на исходном изображении
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Отобразите изображение с контурами
cv2.imshow("Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contour_properties = []

# Для каждого контура найдите координаты Xmax, Xmin, Ymax и Ymin
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    Xmax = x + width
    Xmin = x
    Ymax = y + height
    Ymin = y
    contour_properties.append((Xmax, Xmin, Ymax, Ymin))

    # Нанесите прямоугольники на изображение
    cv2.rectangle(image_main, (Xmin, Ymin), (Xmax, Ymax), (0, 0, 255), 2)

# Отобразите изображение с контурами и прямоугольниками
cv2.imshow("Contours and Rectangles", image_main)
cv2.waitKey(0)
cv2.destroyAllWindows()