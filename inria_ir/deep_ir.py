import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm
import pickle
import numpy as np
import cv2
from sklearn.neighbors import KDTree
import torch
import time


class Feature_Extractor_VGG16:
    
    model = VGG16()
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    for layer in model.layers[:]:
        layer.trainable = False

    def __init__(self) -> None:
        pass

    def extract(self, imgs):
        imgs_shape = imgs.shape
        features = None
        if imgs_shape[1] == 224 and imgs_shape[2] == 224:
            features = self.model.predict(imgs)
        else:
            print('reshape img to (224, 224)')
        return features



class Matching:

    def __init__(self, database) -> None:
        features = database
        s_time = time.time()
        self.kdtree = KDTree(features, leaf_size = 20)
        print(f'[INFO] Build KDTree successfully in {time.time()-s_time}s')

    def query(self, input_query):
        feature = input_query
        s_time = time.time()
        _, indexs = self.kdtree.query(feature.reshape(1, -1), k=7)
        query_time = time.time()-s_time
        print(f'[INFO] Query successfully in {query_time}s')
        return indexs[0], query_time


class Utils:

    def load_img(img_path, shape = (224, 224)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, shape)
        return img


    def load_database(database_path):
        if os.path.exists(database_path):
            print('[INFO] Exists database...')
            with open(database_path, 'rb') as file:
                features, img_paths = pickle.load(file)
        else:

            feature_extractor = Feature_Extractor_VGG16()
            features = np.zeros((1, 4096))
            img_paths = []
            img_dir = './imgs'

            for img_name in tqdm(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, img_name)
                img = Utils.load_img(img_path)
                feature = feature_extractor.extract(img.reshape(1, *img.shape))
                img_paths.append(img_path)
                features = np.concatenate([features, feature.reshape(1, -1)], axis=0)
            features = features[1:]

            print(f'the feature shape = {features.shape}')
            with open(database_path, 'wb') as file:
                pickle.dump([features, img_paths], file)
        return features, img_paths


    def path2id(path_query, paths):
        for i, path in enumerate(paths):
            if path == path_query:
                return i
