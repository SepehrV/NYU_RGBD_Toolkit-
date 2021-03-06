import numpy as np
from scipy import misc
import sys
import os
import ipdb
import h5py
import matplotlib.pyplot as plt


def load_davis(path_images, path_labels, image_size=[240,360], label_size=[120,180]):
    data = {}
    dirs = os.listdir(path_images)
    for d in dirs:
        images = []
        print ('currently at images of : ' + d)
        files = os.listdir(path_images+d)
        files = sorted(files)
        for f in files:
            image = misc.imread(path_images+d+'/'+f)
            images.append(misc.imresize(image, image_size))
        data[d] = [np.array(images)]


    dirs = os.listdir(path_labels)
    for d in dirs:
        labels = []
        print ('currently at labels of : ' + d)
        files = os.listdir(path_labels+d)
        files = sorted(files)
        for f in files:
            image = misc.imread(path_labels+d+'/'+f)
            labels.append(misc.imresize(image, label_size))
        data[d].append(np.array(labels))

    return data

def save_davis(path, data, portion=[0.5,1,1]):
    for subset in data.keys():
        b, h, w, c = data[subset][0].shape
        b_l, h_l, w_l = data[subset][1].shape

        data[subset][0] = data[subset][0].transpose((0,3,1,2)).reshape((b,c,h*w))
        data[subset][1] = data[subset][1].reshape((b_l,h_l*w_l))
        data[subset][1] = np.array(data[subset][1] > 0, dtype='int64')
        train_x = data[subset][0][0:int(b*portion[0])]
        valid_x = data[subset][0][int(b*portion[0]):int(b*portion[1])]
        test_x = data[subset][0][int(b*portion[1]):]
        train_y = data[subset][1][0:int(b*portion[0])]
        valid_y = data[subset][1][int(b*portion[0]):int(b*portion[1])]
        test_y = data[subset][1][int(b*portion[1]):]

        h5f = h5py.File(path + subset, 'w')
        h5f.create_dataset('images_train',data = train_x )
        h5f.create_dataset('images_valid',data = valid_x)
        h5f.create_dataset('images_test',data = test_x)
        h5f.create_dataset('labels_train',data = train_y)
        h5f.create_dataset('labels_valid',data = valid_y)
        h5f.create_dataset('labels_test',data = test_y)
        h5f.close()


path_images = '/usr/data/Datasets/davis-data/JPEGImages/480p/'
path_labels = '/usr/data/Datasets/davis-data/Annotations/480p/'
save_path = '/usr/data/Datasets/davis-data/davis_3/'
data = load_davis(path_images, path_labels)
save_davis(save_path, data)
ipdb.set_trace()
