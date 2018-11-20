# DCGAN-Keras

## Overview

This model generates photo-realistic images by learning image dataset.

The codes are based on [eriklindernoren's repository](https://github.com/eriklindernoren/Keras-GAN/tree/master/dcgan).

You can run this model with [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Files
* main.py : contains model and running part
* utils.py : contains preprocessing function for CelebA dataset & data loader
* haarcascade_frontalface_default.xml : for face recognition (preprocessing)

## Usage
Before you run the model, make directories below.
* `datasets` : put your dataset in this folder
* `logs` : checkpoints are going to be saved here.
* `models` : model architecture and weights are going to be saved here.

You need to edit some hyper parameters in line 23-29 of `main.py`
When you put your own data to `datasets` folder, clarify the resolution of image data by making another folder `*by*`.
> ex) `datasets/CelebA/128by128/*.jpg`

If you don't want to use 128x128 resolution but other, you'd better change the model a bit. If you have any problem to adjust the model to your own data, don't hesitate to *open issue*.

> You need to add code below to `losses.py` file to use [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) as loss function.
~~~
def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
~~~
~~~
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return 100 - 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
~~~

## Sample of result

![e10-i4800](https://user-images.githubusercontent.com/12293076/45536899-c1476e80-b83d-11e8-85fe-7e3295a41d27.png)

