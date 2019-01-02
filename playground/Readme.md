# Playground
Here, I grasp some famous and interesting deep learning projects for me to use and play.

I use Windows 10, 8@i5-8250U CPU, NVIDIA GeForce MX 150 GPU.

1. [neural-style](https://github.com/anishathalye/neural-style)
2.

## neural-style

### Theoretical concepts

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





