#!/usr/bin/python
# -*- coding: utf-8 -*-
import torchvision
import argparse
import os
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Wide_ResNet101_2_Weights
from tqdm import tqdm
from common import (get_pdn_small, get_pdn_medium,
                    ImageFolderWithoutTarget, InfiniteDataloader)

# Этот код определяет функцию "getargparse",
# которая возвращает объект "parser" класса "ArgumentParser"
# из модуля "argparse".
# Сначала создается экземпляр класса "ArgumentParser"
# с указанными параметрами: "prog" - имя программы,
# "description" - описание программы и "epilog" - текст,
# отображаемый внизу помощи.
# Затем, вызывается метод "addargument"
# объекта "parser" для добавления аргумента командной строки.
# В данном случае, добавляется аргумент "-o"
# или "--outputfolder" со значением "default",
# равным 'output/pretraining/1/'.
# Наконец, вызывается метод "parseargs()" объекта "parser"
# для парсинга аргументов командной строки
# и возврата объекта с сохраненными аргументами.
# Таким образом, код создает конфигурацию аргументов командной строки
# и возвращает ее для дальнейшего использования в программе.


def get_argparse():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-o', '--output_folder',
                        default='output/pretraining/1/')
    return parser.parse_args()


# variables
model_size = 'small'
imagenet_train_path = 'C:/Users/User/Downloads/ILSVRC/Data/CLS-LOC/train'
seed = 42
on_gpu = torch.cuda.is_available()
device = 'cuda' if on_gpu else 'cpu'

# constants
out_channels = 384
# grayscale_transform - это преобразование случайной сепии (grayscale)
# с вероятностью 0.1, которое будет одинаково применяться
# к обоим изображениям в дальнейшем.
grayscale_transform = transforms.RandomGrayscale(p=1.0)  # apply same to both
# последовательность преобразований изображения
extractor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# последовательность преобразований изображения
pdn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])


def train_transform(image):
    image = grayscale_transform(image)
    return extractor_transform(image), pdn_transform(image)


def main():
    # устанавливаем seed для генераторов случайных чисел
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Загружает конфигурацию аргументов командной строки
    # с помощью функции get_argparse(),
    # которая была определена ранее,
    # и сохраняет объект конфигурации в переменную config.
    config = get_argparse()

    # Создает папку с заданным именем (config.output_folder)
    # с помощью os.makedirs().
    # Если папка уже существует,
    # то эта команда не делает ничего.
    os.makedirs(config.output_folder)

    # Используя предварительно обученную модель wide_resnet101_2
    # из библиотеки torchvision.models, создает объект backbone.
    # Модель предварительно обучена на наборе данных ImageNet.
    backbone = torchvision.models.wide_resnet101_2(
        weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)

    # Создает объект extractor класса FeatureExtractor,
    # который будет использоваться для извлечения признаков изображений
    # с помощью объекта backbone.
    # Здесь выбираются слои извлечения признаков,
    # устройство (device), на котором будет выполняться извлечение,
    # и форма входных данных.
    extractor = FeatureExtractor(backbone=backbone,
                                 layers_to_extract_from=['layer2', 'layer3'],
                                 device=device,
                                 input_shape=(3, 512, 512))

    # В зависимости от значения переменной model_size,
    # создается модель pdn
    if model_size == 'small':
        pdn = get_pdn_small(out_channels, padding=True)
    elif model_size == 'medium':
        pdn = get_pdn_medium(out_channels, padding=True)
    else:
        raise Exception()

    # извлекаем тренировочные данные
    train_set = ImageFolderWithoutTarget(imagenet_train_path,
                                         transform=train_transform)
    # загружаем их в загрузчик
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                              num_workers=7, pin_memory=True)
    # обеспечиваем возможность бесконечной загрузки данных в модель
    train_loader = InfiniteDataloader(train_loader)

    # находим среднее значение и среднее квадратичное отклонение для обучающей выборке
    channel_mean, channel_std = feature_normalization(extractor=extractor,
                                                      train_loader=train_loader)
    # этот код добавил я
    with open("data.txt", "w") as file:
        # Запись значений переменных в файл
        file.write(f"channel_mean = {channel_mean}\n")
        file.write(f"channel_std = {channel_std}\n")

    pdn.train()
    if on_gpu:
        pdn = pdn.cuda()

    optimizer = torch.optim.Adam(pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    tqdm_obj = tqdm(range(60000))
    # Запускается цикл for, который итерируется по объекту tqdm_obj
    # и перебирает пары значений (image_fe, image_pdn) из train_loader.
    # train_loader - это загрузчик данных, который предоставляет изображения
    # для обучения модели.
    for iteration, (image_fe, image_pdn) in zip(tqdm_obj, train_loader):
        if on_gpu:
            image_fe = image_fe.cuda()
            image_pdn = image_pdn.cuda()

        # вычисляем выход модели extractor на image_fe
        target = extractor.embed(image_fe)
        # нормализация
        target = (target - channel_mean) / channel_std
        # вычисляем выход модели pdn на image_pdn
        prediction = pdn(image_pdn)
        # вычисляем ошибку предсказания между моделями extractor и pdn
        loss = torch.mean((target - prediction)**2)

        # обнуляем градиент оптимизатора
        optimizer.zero_grad()
        # считаем градиенты
        loss.backward()
        # делаем шаг оптимизатором
        optimizer.step()

        # обновляем tqdm_obj отображающее текущее значение функции потерь.
        tqdm_obj.set_description(f'{(loss.item())}')

        # Если номер итерации (iteration) делится на 10000 без остатка,
        # то модель pdn сохраняется в файл teacher_{model_size}_tmp.pth
        # в папке output_folder, а также сохраняется состояние модели pdn
        # в файл teacher_{model_size}_tmp_state.pth.
        if iteration % 10000 == 0:
            torch.save(pdn,
                       os.path.join(config.output_folder,
                                    f'teacher_{model_size}_tmp.pth'))
            torch.save(pdn.state_dict(),
                       os.path.join(config.output_folder,
                                    f'teacher_{model_size}_tmp_state.pth'))

    # По окончании цикла модель pdn сохраняется
    # в файл teacher_{model_size}_final.pth в папке output_folder,
    # а также сохраняется состояние модели pdn в файл teacher...
    torch.save(pdn,
               os.path.join(config.output_folder,
                            f'teacher_{model_size}_final.pth'))
    torch.save(pdn.state_dict(),
               os.path.join(config.output_folder,
                            f'teacher_{model_size}_final_state.pth'))


# функция для нормализации признаков извлекаемых с помощью модели extractor
# - extractor: модель, используемая для извлечения признаков
# - train_loader: загрузчик данных для тренировочной выборки
# - steps: количество шагов,
# на которых будет вычисляться среднее значение и дисперсия признаков
@torch.no_grad()
def feature_normalization(extractor, train_loader, steps=10000):

    mean_outputs = []
    normalization_count = 0
    with tqdm(desc='Computing mean of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            if on_gpu:
                image_fe = image_fe.cuda()
            # Функция первым шагом вычисляет среднее значение признаков
            # (channel_mean). Для этого она проходит
            # через тренировочный загрузчик данных train_loader,
            # извлекает признаки с помощью модели extractor,
            # а затем вычисляет среднее значение
            # по указанным осям (0, 2, 3).
            # Результаты хранятся в списке mean_outputs.
            output = extractor.embed(image_fe)
            mean_output = torch.mean(output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    # объединяем тензоры и вычисляем среднее по каналам
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    # аналогично считаем среднее квадратичное отклонение
    mean_distances = []
    normalization_count = 0
    with tqdm(desc='Computing variance of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            if on_gpu:
                image_fe = image_fe.cuda()
            output = extractor.embed(image_fe)
            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, input_shape):
        super(FeatureExtractor, self).__init__()
        # модель, которая будет использоваться
        # в качестве основы для извлечения признаков
        self.backbone = backbone.to(device)
        # список слоев из модели backbone,
        # из которых будут извлекаться признаки
        self.layers_to_extract_from = layers_to_extract_from
        # тип вычислительного устройства
        self.device = device
        # форма входных данных для модели
        self.input_shape = input_shape
        # экзэмпляр класса PatchMaker, используется для создания
        # патчей на входных данных
        self.patch_maker = PatchMaker(3, stride=1)
        # словарь модулей, используемых для прямого подхода через модель
        self.forward_modules = torch.nn.ModuleDict({})

        # Создание экземпляра класса NetworkFeatureAggregator
        # и присвоение его переменной feature_aggregator.
        # Этот модуль используется для агрегации признаков
        # из различных слоев модели backbone.
        # Он принимает backbone, layers_to_extract_from и device.
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )

        #  Вызов метода feature_dimensions модуля feature_aggregator
        #  для получения размерности признаков.
        #  Размерность возвращается в переменную feature_dimensions.
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # - Создание экземпляра класса Preprocessing
        # и присвоение его переменной preprocessing.
        # Этот модуль используется для проведения
        # предварительной обработки (preprocessing) признаков
        # перед агрегацией.
        # Он принимает размерность признаков feature_dimensions
        # и фиксированное значение 1024.
        preprocessing = Preprocessing(feature_dimensions, 1024)
        self.forward_modules["preprocessing"] = preprocessing

        # Создание экземпляра класса Aggregator
        # и присвоение его переменной preadapt_aggregator.
        # Этот модуль используется для агрегации признаков
        # после предварительной обработки.
        preadapt_aggregator = Aggregator(target_dim=out_channels)

        # Перемещение модуля preadapt_aggregator на заданное устройство device.
        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # установка всех модулей в режим оценки
        self.forward_modules.eval()

    @torch.no_grad()
    def embed(self, images):
        """Returns feature embeddings for images."""
        # устанавливаем режим оценки для объекта "feature_aggregator"
        # из словаря "forward_modules"
        _ = self.forward_modules["feature_aggregator"].eval()
        # используем метод "feature_aggregator" для вычисления признаков изображения
        features = self.forward_modules["feature_aggregator"](images)

        # извлекаем определенные слои из вычесленных признаков
        # на основе значения атрибута self.layers_to_extract_from
        features = [features[layer] for layer in self.layers_to_extract_from]

        # Четвертая строка использует объект "patchmaker"
        # для разбиения каждого извлеченного признака на патчи (сегменты)
        # и возвращает также информацию о пространственной локации
        # каждого патча. Полученные разбитые признаки сохраняются
        # в списке "features", а формы патчей сохраняются
        # в списке "patchshapes".
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in
            features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        # сохраняем кол-во патчей
        # первого разбитого признака в переменную
        # ref_num_patches
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # изменяется с использованием метода reshape
            # он изменяет форму _features, чтобы соответсвовать
            # размерам патчей
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1],
                *_features.shape[2:]
            )
            # меняем порядок осей в тензоре
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            # сохраняем порядок
            perm_base_shape = _features.shape
            # снова изменяем размер
            _features = _features.reshape(-1, *_features.shape[-2:])
            # Метод F.interpolate используется для интерполяции
            # _features в новый размер.
            # Если _features имеет размер (H, W),
            # то после интерполяции они становятся равными
            # (ref_num_patches[0], ref_num_patches[1]).
            # Здесь используется режим "bilinear" и align_corners=False.
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )

            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1,
                                          *_features.shape[-3:])
            features[i] = _features
        # Все фичи в списке features изменяются
        # с использованием метода reshape для приведения
        # их к одному размеру.
        # Каждый элемент списка features изменяется
        # на x.reshape(-1, *x.shape[-3:]).
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        # Модуль preprocessing из forward_modules применяется к features.
        features = self.forward_modules["preprocessing"](features)
        # Модуль preadapt_aggregator из forward_modules применяется к features.
        features = self.forward_modules["preadapt_aggregator"](features)
        # features изменяется в форму (-1, 64, 64, out_channels).
        features = torch.reshape(features, (-1, 64, 64, out_channels))
        # Оси features переставляются с использованием метода permute.
        # Порядок осей изменяется на (0, 3, 1, 2).
        features = torch.permute(features, (0, 3, 1, 2))

        return features


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    # Метод patchify принимает входные данные
    # features (тензор размером bs x c x w x h)
    # и возвращает тензор патчей соответствующего размера.
    # Если параметр return_spatial_info установлен в True,
    # то метод возвращает также информацию
    # о пространственном расположении патчей.

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        # В начале метода вычисляется значение padding
        # (целочисленное деление (patchsize - 1) на 2)
        padding = int((self.patchsize - 1) / 2)
        # создается объект unfolder класса torch.nn.Unfold.
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding,
            dilation=1
        )
        # Далее, данный объект unfolder
        # используется для развертки features в патчи методом вызова
        # unfolder(features). Результат развертки записывается в
        # unfolded_features.
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        # Затем метод вычисляет общее количество патчей
        # для каждого измерения [w, h] тензора features.
        # Это число вычисляется по формуле
        # (s + 2 * padding - 1 * (patchsize - 1) - 1) / stride + 1,
        # где s - размер соответствующего измерения.
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        # Затем unfolded_features транспонируется с помощью метода permute,
        # чтобы изменить порядок осей на [bs, w//stride, h//stride, c, patchsize, patchsize]
        # и вернуть итоговую матрицу патчей.
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        # Если параметр return_spatial_info установлен в True,
        # то метод возвращает матрицу патчей и
        # количество патчей для каждого измерения [w, h].
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


# Определение конструктора класса Preprocessing,
# который принимает аргументы input_dims (размер входных данных)
# и output_dim (размер выходных данных).
class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        # Создание списка модулей для предобработки данных.
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            # Создание экземпляра модуля MeanMapper
            # измерению заданного размера output_dim.
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    # Определение метода forward,
    # который принимает аргумент features (входные данные).
    def forward(self, features):
        # создание пустого списка, который будет хранить результаты предобработаки
        _features = []
        # Выполнение итераций по списку модулей для предобработки данных и входным данным.
        for module, feature in zip(self.preprocessing_modules, features):
            # Вызов метода __call__ модуля module
            # для обработки входных данных feature
            # и добавление результата в список _features.
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


# Этот код представляет собой определение класса MeanMapper,
# который является наследником класса torch.nn.Module.
# Код реализует модель, которая выполняет предварительную обработку
# данных путем вычисления среднего значения по каждому признаку.
class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features,
                                     self.preprocessing_dim).squeeze(1)

# Этот код представляет собой определение класса Aggregator,
# который является наследником класса torch.nn.Module.
# Код реализует модель, которая выполняет
# агрегацию признаков путем изменения их формы
# и вычисления среднего значения.


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        # список строк, предсталяющих слои, из которых следует
        # извлекать признаки
        self.layers_to_extract_from = layers_to_extract_from
        # предтренированная сеть
        self.backbone = backbone
        self.device = device
        # Если у backbone нет атрибута hook_handles,
        # то он добавляется в виде пустого списка для отслеживания хэндлов.
        # Затем, для удаления возможных конфликтующих хуков,
        # удаляются все существующие хэндлы хуков из backbone.
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        # Атрибут outputs инициализируется пустым словарем
        # для хранения извлеченных признаков сети.
        self.outputs = {}

        # затем, для каждого слоя, казанного в
        # layers_to_extract_from
        for extract_layer in layers_to_extract_from:
            # Создается экземпляр ForwardHook с параметрами outputs,
            # extract_layer и последним слоем
            # из layers_to_extract_from.
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            # - Если в строке extract_layer есть символ точки (.),
            # это означает, что слой находится внутри подмодуля backbone.
            # - Код разделяет строку extract_layer по символу точки,
            # чтобы получить имя подмодуля и индекс слоя.
            # - Затем из backbone.dict["_modules"]
            # извлекается соответствующий подмодуль.
            # - Если extract_idx является числом,
            # он преобразуется в целое число,
            # а затем из полученного подмодуля
            # извлекается соответствующий слой.
            # - Если extract_idx не является числом,
            # а из словаря внутри подмодуля извлекается
            # соответствующий подмодуль,
            # который присваивается переменной network_layer.
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][
                        extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]
            # - Если network_layer является экземпляром torch.nn.Sequential,
            # на последний слой последовательного модуля регистрируется хук
            # с помощью register_forward_hook.
            # - Если network_layer не является последовательным модулем,
            # на сам слой network_layer регистрируется хук.
            # - Хэндл хука добавляется в список backbone.hook_handles.
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        # экземпляр NetworkFeatureAggregator
        # перемещается на указанное устройство device.
        self.to(self.device)

    # Сначала список self.outputs очищается.
    # Затем идет блок кода,
    # обернутый в контекстный менеджер torch.no_grad().
    # Контекстный менеджер torch.no_grad()
    # используется для отключения вычисления градиентов
    # во время прямого прохода.
    # Это полезно, если вы только хотите использовать модель
    # для инференса, без необходимости обновлять параметры модели.
    # Внутри блока кода выполняется вызов self.backbone(images).
    # self.backbone представляет собой некоторую модель нейронной сети,
    # которая принимает на вход images и вычисляет фичи модели.
    # Этот вызов может вызвать исключение с именем
    # LastLayerToExtractReachedException, которое указывает на то,
    # что модель достигла последнего слоя для вычисления фичей.
    # Если такое исключение возникло, то блок кода игнорирует его
    # и продолжает выполнение.
    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    # В данном коде определена функция feature_dimensions,
    # которая вычисляет размерности фичей
    # для всех слоев нейросети на основе заданной входной формы
    # input_shape.
    # Сначала создается входной тензор _input размером
    # 1 + input_shape, где 1 указывает на размерность пакета
    # (batch size) равную 1.
    # Затем этот тензор перемещается на то же устройство,
    # на котором находится экземпляр класса.
    # Затем вызывается модель self
    # (которая представляет собой некоторую нейросеть)
    # с _input в качестве входа.
    # Это приводит к вычислению предсказаний модели для _input
    # и сохранению этих предсказаний в _output.
    # Затем происходит итерация по слоям
    # (заданным в атрибуте layers_to_extract_from)
    # и для каждого слоя вычисляется размерность фичей
    # (количество каналов) в соответствующем элементе _output.
    # Результаты сохраняются в список, который затем возвращается
    # в качестве результата функции feature_dimensions.
    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in
                self.layers_to_extract_from]

# В данном коде определен класс ForwardHook,
# который используется для создания "хуков" (hooks)
# в нейронной сети для извлечения промежуточных фичей.
#
# Класс ForwardHook имеет два метода: init и call.
# Метод init инициализирует объект ForwardHook
# и сохраняет переданные значения hook_dict,
# layer_name и last_layer_to_extract. hook_dict
# представляет собой словарь,
# в который будут сохраняться выходы слоев,
# layer_name - имя текущего слоя,
# а last_layer_to_extract - имя последнего слоя,
# до которого нужно сохранять выходы.
# Если layer_name совпадает с last_layer_to_extract,
# то создается глубокая копия True,
# которая будет использоваться для вызова исключения
# LastLayerToExtractReachedException.
# Метод call вызывается при прямом проходе
# через слой нейросети.
# Он сохраняет выход слоя в словаре под
# соответствующим ключом layer_name.
# Затем проверяется, совпадает ли layer_name
# с last_layer_to_extract.
# Если совпадает, вызывается исключение
# LastLayerToExtractReachedException. В конце метод возвращает None.
# Использование этого класса позволяет добавить
# "хуки" в нейронную сеть для отслеживания выходов
# определенных слоев и остановки прямого прохода
# на заданном слое. Это может быть полезно,
# если нужно извлечь промежуточные фичи
# до определенного слоя для дальнейшего анализа.


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None

# В данном коде определен класс LastLayerToExtractReachedException,
# который является подклассом встроенного класса Exception.
# Класс LastLayerToExtractReachedException
# позволяет определить своё собственное исключение,
# которое можно вызвать внутри кода. В данном случае,
# класс не содержит никакой дополнительной логики или атрибутов,
# и просто служит для обозначения случая,
# когда достигнут последний слой, до которого нужно сохранять выходы.
# Вызов этого исключения позволяет контролировать
# прямой проход через нейронную сеть
# и остановить его на нужном слое. Это может быть полезно,
# если необходимо сохранить промежуточные фичи
# только до определенного слоя и больше оставаться не нужно.


class LastLayerToExtractReachedException(Exception):
    pass


if __name__ == '__main__':
    main()
