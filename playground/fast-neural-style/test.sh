#!/usr/bin/env bash


cmd="python evaluate.py --checkpoint ckpts/$2.ckpt --in-path contents/$1.jpg --out-path outputs/$1_$2.jpg"
echo $cmd
${cmd}


