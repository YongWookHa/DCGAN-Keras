import os
from time import localtime, strftime

from glob import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from scipy.misc import imread, imresize
import numpy as np
import cv2


def time():
    return strftime('%m%d_%H%M', localtime())

def crop_face(dataPath, savePath, cascPath='haarcascade_frontalface_default.xml'):
    # find a face in image and crop it
    faceCascade = cv2.CascadeClassifier(cascPath)
    li = os.listdir(dataPath)
    count = 0

    # Read the image
    for fn in li:

        image = cv2.imread(os.path.join(dataPath, fn))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 5, 5)

        if len(faces) != 1:
            pass
        else:
            x, y, w, h = faces[0]
            image_crop = image[y: y+w, x : x+w, :]
            image_resize = cv2.resize(image_crop, (64, 64))
            cv2.imwrite(os.path.join(savePath, fn), image_resize)
            print(fn)
            count+=1

    print("total: %d / %d" % (count, len(li)))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

class DataLoader():

    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def get_n_data(self):
        dirs = os.listdir('./datasets/%s/%sby%s/' % (self.dataset_name, self.img_res[0], self.img_res[1]))
        return len(dirs)

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%sby%s/*' % (self.dataset_name, self.img_res[0], self.img_res[1]))

        batch_images = np.random.choice(path, size=batch_size)

        ref_imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            h, w = self.img_res
            img = imresize(img, self.img_res)  # for using vgg network

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)

            ref_imgs.append(img)

        ref_imgs = np.array(ref_imgs) / 127.5 - 1.

        return ref_imgs

    def imread(self, path):
        return imread(path, mode='RGB').astype(np.float)
