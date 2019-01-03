# Playground
Here, I grasp some famous and interesting deep learning projects for me to use and play.

I use Windows 10, 8@i5-8250U CPU, NVIDIA GeForce MX 150 GPU.

1. [neural-style](https://github.com/anishathalye/neural-style)
2. [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)
3. shell

## Neural-style

### Usage
`cd C:\Users\kanny\Desktop\playground\neural-style`

`python neural_style.py --content contents\fhr.jpg --styles styles\sky100k.jpg --output outputs\fhr_sky100k.jpg 
--iterations 2000 --checkpoint-output checkpoints\fhr_sky100k_%s.jpg --checkpoint-iterations 100 
--network networks\imagenet-vgg-verydeep-19.mat --overwrite`

Run `python neural_style.py --help` to see a list of all options.
Here are some important params:

1. `--checkpoint-output` creates intermediate outputs after each 100 iterations to check it in progress and terminate earlier if the result is good enough.
2. `--content-weight = 5e0` & `--style-weight = 5e2`. If we higher the content-weight or lower the style-weight, then less style, more content, less abstract.
3. `LEARNING_RATE = 1e1`.
4. `--style-layer-weight-exp=1`. tweak how "abstract" the style transfer should be. Lower values mean that 
style transfer of a finer features will be favored over style transfer of a more coarse features. Default value is 1.0 - all layers treated equally.
(note that we use several layers VGG filters. Maybe finer features is more important). **Lower (eg: 0.2) is less abstract (see the content clearer)**.
5. `--pooling=max`. avg is less abstract.
6. `--preserve-colors=FALSE`. If TURE, adds post-processing step, which combines colors from the original image and luma from the stylized image (YCbCr color space), thus producing color-preserving style transfer.
If TRUE, only heritage texture and luma of style figure.
7. `--overwrite` overwrite the existed output figure.

### Results

1. Size: content: 12k, style: 100k, iter: 1000. Time: 400s
2. Size: content: 12k, style: 500k, iter: 1000. Time: 400s
3. Size: content: 250k, style: 500k, iter: 1000. Time: 600s
4. Size: content: 600k, style: 600k, out of memory for my computer

### Theoretical concepts

The weights are not trained here, we use pretrained vggnet weights. What we have is content figure and style figure and what we want is output figure. At first the output figure is noise (or content figure to accelerate speed). We learn the output figure each time.

We put content, style, output in vggnet and can get several features (eg: conv1_2). Although these three figures are different in size, we can still use vggnet (because conv layer don't care about the input_size). conv1_2(content) is a cube.

We set **content loss as the MSE of conv1_2(content) and conv1_2(output). style loss is MSE of G(style) and G(output), while G is the GRAM matrix (covariance without centered)**.

![](.././pics/vgg.jpg)

In right fig, y is the data (input) in left fig. Note that relu1_2 is acutally a cube (64 dimension rather than 3 dim as in a orinary figure), but we can plot it. From the right fig, we can see indeed that VGG learns the texture of a figure.

We can also realize it ourselves with the help of [Pytorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
However, there are many tricks and strange codes in it. Thus I don't pay time looking at it.

## Fast-style-transfer
### Usage
#### Requirement
`pip install --trusted-host pypi.python.org moviepy`

type `python`, `import imageio` then `imageio.plugins.ffmpeg.download()`. See [here](https://github.com/lengstrom/fast-style-transfer/issues/129)

#### Evaluation
We have downloaded (learned) sevral `ckpt` files (in ckpts folder) which we can use directly. In styles folder, we can see what these styles are. These ckpt files is the learned weights of Transform Net.

`cd C:\Users\kanny\Desktop\playground\fast-style-transfer`

`python evaluate.py --checkpoint ckpts\la_muse.ckpt --in-path contents\fhr.jpg --out-path outputs\fhr_la_muse.jpg`

One figure in 3 seconds. Fast, nearly real-time.

Also, there is a [website](https://tenso.rs/demos/fast-neural-style/) for us to run the evaluation online.

#### Train
If I want to train my own style, I first need to download a 12GB file COCO dataset (figures) and then train for around 4 month (but 4 hours on a Maxwell Titan X).

`python style.py --style styles\shinkai0.jpg --checkpoint-dir checkpoints --test contents\building.jpg --test-dir test_directory --content-weight 1.5e1 --checkpoint-iterations 100 --batch-size 1`

Too long a time for my poor laptop.

### Theoretical concepts
Rather than learning input which is a generative model, we turn into a supervised model (learning weights). Thus, it takes more time in training step but is real-time in test step.

First, we fix a style figure. The input is a figure (from train 2014 dataset). Each time we put input figure into a Transform Net, we get output of the same size (also is a 3D cube figure). The weights of Transform Net is learnable. **We treat Transform_net_output as the desired output.** Now is the same as neural-style model: we put output, content, style into VGG and get LOSS function. We update weights to minimize the LOSS.

Thus, for every style, we learn a Transform Net. (If input fixed, the loss should also fixed)

Transform Net:

1. Eschew pooling layer and use fractionally strided conv layer. Reason: conv layer is more smooth and can learn its own function (by diferrent weight).
2. Use residual blocks. Reason: residual blocks can learn identify function (we indeed want some identifications between output and input).
3. All conv layers are followed by batch normalization + RELUE.
4. downsample(stride 2) + upsample(stride 1/2) in conv layer. Reason: save computations (compared with conv layer of stride 1). have deeper receptive field sizes.
5. In the last layer, we use scaled tanh to turn the output in [0,255].
5. Use L-BFGS (qausi-Newton) to update. Also, we can use Adam, which is faster.

LOSS = content + style + TV (Total Variation Regularization).

We can see this model in a easy way. If we forget VGG net, we use a Network to get output and compute LOSS (**very specific deep learning unreasonable thoughts**): pixel-difference of output and style. However, pixel-difference cannot handel semantic and Perceptral things (eg: if move a figure one pixel up, pixel loss is high). Luckily VGG can represent some Perceptral of a figure.

![](.././pics/fast_style_structure.jpg)

In left fig, input figure is the same as content figure. In middle fig is the structure of Transform Net. In right fig is the residual block.

## Shell
Shell (bash) is really useful and convenient. However I am using windows... 

First I download [Cygwin](https://www.cygwin.com/). In the installatin, I choose to add `bash`. Finally, I get a desktop: Cygwin64 Terminal.

Now I can use `pwd` `ls` `mkdir` `echo`.

`cd /cygdrive/c/Users/kanny/Desktop/playground/shell`

`bash test.sh`

We can choose different arguments.

`cd /cygdrive/c/Users/kanny/Desktop/playground/fast-style-transfer`

`bash test.sh fhr la_muse`

Also, we can use `do` as for loop. See `test2.sh` in fast-style-transfer directory.
