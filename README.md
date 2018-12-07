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
* `logs` : checkpoints are going to be saved here
* `models` : model architecture and weights are going to be saved here

---

### main.py
You need to edit some hyper parameters in line 23-29 of `main.py`.
When you put your own data to `datasets` folder, clarify the resolution of image data by making another folder `*by*`. 
*ex)* `datasets/CelebA/128by128/*.jpg`

> If you don't want to use 128x128 resolution but other, you'd better change the model a bit. If you have any problem to adjust the model to your own data, don't hesitate to *open issue*.

### utils.py
You can crop faces from CelebA dataset with `util.py`. (the image file in CelebA should be `jpg`)

Import this file to console and call `crop_face` function.

The function has 4 parameters.
* `dataPath` : path of face images
* `savePath` : path to save the cropped image
* `target_size` : need to be tuple *ex) (128, 128)*
* `cascPath` : haar-cascade xml file path

The base code of class `DataLoader` is from (eriklindernoren/Keras-GAN)[https://github.com/eriklindernoren/Keras-GAN].
I recommend to visit his repo, because there are so many good example code for GAN. :)

Anyway, there are two ways to load data with `DataLoader`.
* Load all the data into RAM. : this would be way faster when you have enough RAM.
* Randomly picked image data from the designated directory. : though it's slower, this way doesn't need much RAM space.

## Sample of result
* after 10 epoch <br/>
![e10-i4800](https://user-images.githubusercontent.com/12293076/45536899-c1476e80-b83d-11e8-85fe-7e3295a41d27.png)

