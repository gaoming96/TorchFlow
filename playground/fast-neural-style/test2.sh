#!/usr/bin/env bash

text="usage: ./generate_video.sh \"path_to_src_frames\" \"path_to_style\" \"path_to_output_dir\"\nfor example:\n./generate_video.sh \"../video/vid_1/\*.png\" \"models/composition.model\" \"../out/chainer/video/vid_1/\"\n\n"
echo -e $text

if [ $# != 3 ]
   then
   echo "not 3 parameters specified, terminating"
   exit
fi

for i in $1;
do
   file=`basename $i`
   cmd="python generate.py $i -m $2 -o $3/$file -g 0"
   echo $cmd
   ${cmd}
done