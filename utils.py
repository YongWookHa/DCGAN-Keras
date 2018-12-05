import os
from time import localtime, strftime

from glob import glob
from scipy.misc import imread, imresize
import numpy as np
import cv2


def time():
    return strftime('%m%d_%H%M', localtime())

def crop_face(dataPath, savePath, target_size, cascPath='haarcascade_frontalface_default.xml'):
    # find a face in image and crop it
    faceCascade = cv2.CascadeClassifier(cascPath)
    li = os.listdir(dataPath)
    count = 0

    # Read the image
    for fn in li:
        if fn[-4:] == '.jpg':
            image = cv2.imread(os.path.join(dataPath, fn))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 5, 5)

            if len(faces) != 1:
                pass
            else:
                x, y, w, h = faces[0]
                image_crop = image[y: y+w, x : x+w, :]
                image_resize = cv2.resize(image_crop, target_size)
                cv2.imwrite(os.path.join(savePath, fn), image_resize)
                print(fn)
                count+=1
        else:
            pass

    print("total: %d / %d" % (count, len(li)))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128), mem_load=False):

        self.mem_load = mem_load
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.path = glob('./datasets/%s/%sby%s/*.jpg' % (self.dataset_name, self.img_res[0], self.img_res[1]))
        self.n_data = len(self.path)
        if self.mem_load:
            self.total_imgs = np.array(list(map(self.imread, self.path))) / 127.5 -1.

    def load_data(self, batch_size=1, is_testing=False):
        imgs = [] # images to be returned
        if self.mem_load:
            idx = np.random.choice(range(self.n_data), size=batch_size)
            for i in idx:
                imgs.append(self.total_imgs[i])
            imgs = np.array(imgs)
        else:
            batch_images = np.random.choice(self.path, size=batch_size)

            for img_path in batch_images:
                img = self.imread(img_path)
                # If training => do random flip
                # if not is_testing and np.random.random() < 0.5:
                    # img = np.fliplr(img)
                imgs.append(img)

            imgs = np.array(imgs) / 127.5 - 1.
        return imgs

    def get_n_data(self):
        return self.n_data

    def imread(self, path):
        return imread(path, mode='RGB').astype(np.float)
