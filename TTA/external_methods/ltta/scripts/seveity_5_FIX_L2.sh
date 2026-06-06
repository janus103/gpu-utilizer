#! /bin/bash

VAL_PY='validate_dwt.py'
NUM_CLS='10'
KERNEL_SIZE='3 3 3'
D_LEV='2 2 2'
KERNEL_FIX='1'
MODEL='resnet26_dwt_img'
WEIGHT='0'
BN_OPT='2'
# BN_OPT = '0': Normal
# BN_OPT = '1': Statistic Entropy
# BN_OPT = '2': Statistic Weight of frequency Domain 

CUDA_VISIBLE_DEVICES=$1 $VAL_PY data/cifar10 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 1 &&

CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/gaussian_noise_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/shot_noise_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/impulse_noise_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&

CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/defocus_blur_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/glass_blur_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/motion_blur_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/zoom_blur_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&

CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/snow_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/frost_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/fog_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/brightness_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&

CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/contrast_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/elastic_transform_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/pixelate_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0 &&
CUDA_VISIBLE_DEVICES=$1 $VAL_PY /home/oem/servers/datasets/classification/new/test-C-10/jpeg_compression_5 --model $MODEL -b 128 -j 5 --input-size 3 224 224 --checkpoint $2 --num-classes $NUM_CLS --dwt-kernel-size $KERNEL_SIZE --dwt_level $D_LEV --dwt_bn $KERNEL_FIX $BN_OPT 0 --no-prefetcher --weight_net $WEIGHT --results-file 0