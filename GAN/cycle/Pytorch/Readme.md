# CycleGAN_Pytorch
Pytorch implementation of CycleGAN.
Based on LynnHo/CycleGAN-Tensorflow-PyTorch/tree/master/pytorch and the original paper, but is concise.

## Download Datasets
- Download the horse2zebra dataset:
```bash
sh ./download_dataset.sh horse2zebra
```
- Download the apple2orange dataset:
```bash
sh ./download_dataset.sh apple2orange
```
- See download_dataset.sh for more datasets

Note that we need to change the datasets in order to use `torchvision.datasets.ImageFolder`. 

Let's say I am going to use ImageFolder("/train/") to read jpg files in folder train.

The file structure is

/train/

-- 1.jpg

-- 2.jpg

-- 3.jpg

I failed to load them, leading to errors:

`RuntimeError: Found 0 images in subfolders of: ./data. Supported image extensions are: .jpg,.JPG,.jpeg,.JPEG,.png,.PNG,.ppm,.PPM,.bmp,.BMP`


I read the solution above and tried tens of times. When I changed the structure to

/train/1/

-- 1.jpg

-- 2.jpg

-- 3.jpg

But the read in code is still -- ImageFolder("/train/"), IT WORKS.

It seems like the program tends to recursively read in files, that is convenient in some cases.

See - https://github.com/pytorch/examples/issues/236.

## Train
