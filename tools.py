import numpy as np
from scipy import misc
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt

import os
import re
from collections import OrderedDict

import pdb


class Interpolate(object):
    """Docstring for interpolate. """
    def __init__(self, rgb_images, labels, sort=True):
        """
        rgb_images: a dict contating names of files (subset/name) and
            RGB images.
        labels: a dict containg names of files (subset/name) and
            corresponding label
        """
        if sort:
            self.images = self.sort_frames(rgb_images)
        else:
            self.images = rgb_images
        self.labels = labels
        #self.getOpticalFlow()
        #self.merge()
        #self.genLabels()


    def sort_frames(self, data):
        """
        sorts the data chronically.
        data: dictionary where keys are file names (subset/r- ... .ppm)
            and values are images
        """
        out = OrderedDict()
        time_stamps = {}
        for name in data.keys():
            t = re.findall('\d{10}\.\d{6}', name)[0]
            time_stamps[name] = float(t)
        times = time_stamps.values()
        times.sort()
        for t in times:
            for k, v in time_stamps.items():
                if v == t:
                    out[k] = data[k]
        return out

    def getOpticalFlow(self):
        """
        computes optical flow of self.images and store them in
            self.flow.
        """
        self.flows = OrderedDict()
        for i in range(len(self.images.keys())-1):
            flow = cv2.calcOpticalFlowFarneback(self.images.values()[i].mean(axis=2),
                                                self.images.values()[i+1].mean(axis=2),
                                                pyr_scale=0.5, levels=1,
                                                winsize=17, iterations=30,
                                                poly_n=7, poly_sigma=1.5,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            self.flows[self.images.keys()[i]] = flow
        self.flows[self.images.keys()[i+1]] = flow

    def merge(self):
        """
        merges all the data into one dict.
        keys are names of the files and values are:
            dict: {image:Image, flow:Flow, label:Label}
        """
        self.merged = OrderedDict()
        for name in self.images.keys():
            self.merged[name] = {'image':self.images[name],
                                 'flow': self.flows[name],
                                 'label': []}

    def gen_dataset(self, n_images=20, both_direction=True):
        """
        generate a subdata from each subset where each sample is consist of
            sequence of images with 1 label corresponding to the last frame.
        n_images: number of images preceding the labelled image
        both_direction: if to use images from both direction to get twice as
            much samples
        """
        m = len(self.images.keys())
        samples_X = []
        samples_Y = []

        for name in self.labels.keys():
            c_index = self.images.keys().index(name)
            try:
                x = []
                for i in range(c_index-n_images, c_index):
                    x.append(self.images.values()[i])
                samples_X.append(x)
                samples_Y.append(self.labels[name])
            except IndexError:
                pass

            if both_direction:
                try:
                    x = []
                    for i in range(c_index, c_index+n_images)[::-1]:
                        x.append(self.images.values()[i])
                    samples_X.append(x)
                    samples_Y.append(self.labels[name])
                except IndexError:
                    pass

        return samples_X, samples_Y


    def genLabels(self):
        """
        generate labels for all images using sparsly annotated images.
        """
        m = len(self.merged.keys())
        sum_flows = [np.zeros(self.labels.values()[0].shape+(2,))]*m
        for i in range(m):
            c_up = -m
            c_down = -m

            for up in range(i, m):
                if self.merged.keys()[up] in self.labels.keys():
                    label_up = self.labels[self.merged.keys()[up]]
                    c_up = up
                    break

            for down in range(i)[::-1]:
                if self.merged.keys()[down] in self.labels.keys():
                    label_down = self.labels[self.merged.keys()[down]]
                    c_down = down
                    break

            if abs(c_down-i) < abs(c_up-i):
                self.merged.values()[i]['label'] = label_down
                c_index = c_down
            else:
                self.merged.values()[i]['label'] = label_up
                c_index = c_up

            #flow = np.zeros(self.merged.values()[i]['label'])
            if c_index == i:
                continue
            if c_index < i:
                Range = range(i, c_index)
            else:
                Range = range(i, c_index)[::-1]

            for j in Range:
                sum_flows[i] = sum_flows[i] + self.merged.values()[j]['flow']
            sum_flows[i] = np.sign(i-c_index)*sum_flows[i]

        for i in range(m):
            self.merged.values()[i]['flow'] = sum_flows[i]
            self.merged.values()[i]['label'] = self.moveLabel(
                                self.merged.values()[i]['label'],
                                self.merged.values()[i]['flow'])

    def moveLabel(self, label, flow):
        """
        move the label according to the flow to gen new label.
        label: uint16 annotated image
        flow: the summed flow from label to the true position of the label
        """
        unknown_label = 0
        m, n = label.shape
        out = np.zeros(label.shape)
        x = np.asarray(np.vstack([range(m)]*n)).T
        y = np.asarray(np.vstack([range(n)]*m))
        mesh = np.asarray([x,y]).transpose((1,2,0))
        mesh = mesh - flow
        for i in range(m):
            for j in range(n):
                try:
                    h, w = mesh[i,j]
                    out[i,j] = label[(int(h), int(w))]
                except IndexError:
                    out[i,j] = unknown_label
        return out

    def show(self):
        """
        shows images and labels from self.merged
        """
        i = 0
        for name, data in self.merged.items():
            i = i+1

            if i%20 == 0:
                f = plt.figure()
                f.add_subplot(2,1,1)
                plt.imshow(data['image'])
                f.add_subplot(2,1,2)
                plt.imshow(data['label'])
                plt.show()


def load_save_all(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:






def load_chunk(path, subset, downsample=None):
    """
    loads all rgb images from a subset of nyu rgbd dataset.
    returns names and images.
    path: path to the dir containing nyu dataset.
    subset: name of the subset contaiing images
    downsample: a tuple indicating new image size
    """
    rootdir = path
    for root, dirs, files in os.walk(rootdir):
        for d in dirs:
            if d == subset:
                subset_path = root+d+'/'
                break

    names = os.listdir(subset_path)
    data = {}
    for name in names:
        if name.split('.')[-1] == 'ppm':
            image = misc.imread(subset_path+name)
            if image.ndim != 3:
                continue
            if downsample != None:
                image = misc.imresize(image,downsample)
            data[subset+'/'+name] = image
    return data

def load_labels(path, subset=None, downsample=None):
    """
    loads labels and names.
    path: path to labels mat file.Containing both 'labeles' and corresponding
        rgb 'file_names'.
    subset: if None, will return all the labels. if set, will only return
        labels corresponding to subset
    downsample: a tuple indicating new image size
    """
    data = sio.loadmat(path)
    out = {}
    for i in range(len(data['file_names'])):
        name = data['file_names'][i]
        name = re.findall("\w*\/r-\S*.ppm", str(name))[0]
        if subset != None:
            if subset in name:
                image = data['labels'][:,:,i]
                if downsample != None:
                    image = cv2.resize(image, downsample[::-1], interpolation=cv2.INTER_NEAREST)
                    #image = misc.imresize(image,downsample, interp='nearest')
                out[name] = image
        else:
            image = data['labels'][:,:,i]
            if downsample != None:
                image = misc.imresize(image,downsample)
            out[name] = image

    return out


def main():
    path_nyu = '/usr/data/Datasets/NYU-RGBD/dining_rooms_part1/'
    subset = 'dining_room_0009'
    path_images = '/usr/data/Datasets/NYU-RGBD/dining_rooms_part1/dining_room_0009/'
    path_labels = '/usr/data/Datasets/NYU-RGBD/labels.mat'
    downsample = (240,320)

    labels = load_labels(path_labels, subset=subset, downsample=downsample)
    interpolate = Interpolate(load_chunk(path_nyu, subset, downsample=downsample), labels)
    samples_x, samples_y = interpolate.gen_dataset()
    pdb.set_trace()
    #interpolate.show()

if __name__ == '__main__':
    main()

