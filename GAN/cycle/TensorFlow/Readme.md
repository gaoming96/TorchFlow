# CycleGAN_Pytorch
GPU Pytorch implementation of CycleGAN.
Based on architrathore/CycleGAN and the original paper, but is concise.

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

## Train & test
- `layers.py` combines a brunch of elemental layers into `general_conv2d` and `general_deconv2d`.
- `models.py` builds several higher structures such as `build_generator_resnet_6blocks` and `build_gen_discriminator` using `layers.py`.
- `main.py` sets input, placeholder, model, loss, optimizer.minimize, saves training images, tf.Session, and tests.

Note that the reuse procedure is different from Pytorch version.

### LOSS
![](../../.././pics/cycle_gan_structure1.png)
