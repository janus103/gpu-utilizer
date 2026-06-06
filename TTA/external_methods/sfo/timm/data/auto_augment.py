""" AutoAugment, RandAugment, AugMix, and 3-Augment for PyTorch

This code implements the searched ImageNet policies with various tweaks and improvements and
does not include any of the search code.

AA and RA Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

AugMix adapted from:
    https://github.com/google-research/augmix

3-Augment based on: https://github.com/facebookresearch/deit/blob/main/README_revenge.md

Papers:
    AutoAugment: Learning Augmentation Policies from Data - https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection - https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty - https://arxiv.org/abs/1912.02781
    3-Augment: DeiT III: Revenge of the ViT - https://arxiv.org/abs/2204.07118

Hacked together by / Copyright 2019, Ross Wightman
"""
import random
import math
import re
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageOps, ImageEnhance, ImageChops, ImageFilter
import PIL
import numpy as np


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

_LEVEL_DENOM = 10.  # denominator for conversion from 'Mx' magnitude scale to fractional aug level for op arguments

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

if hasattr(Image, "Resampling"):
    _RANDOM_INTERPOLATION = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.Resampling.BICUBIC
else:
    _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.BICUBIC


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', _DEFAULT_INTERPOLATION)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    if _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)

    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)

    return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def gaussian_blur(img, factor, **__):
    img = img.filter(ImageFilter.GaussianBlur(radius=factor))
    return img


def gaussian_blur_rand(img, factor, **__):
    radius_min = 0.1
    radius_max = 2.0
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(radius_min, radius_max * factor)))
    return img


def intensity(img, factor, **__):
    """Combined brightness and contrast adjustment.
    
    Applies the same enhancement factor to both brightness and contrast.
    This is useful for treating "intensity/luminance" as a single augmentation type.
    
    Args:
        img: PIL Image to augment.
        factor: Enhancement factor (1.0 = no change, <1.0 = darker/lower contrast, >1.0 = brighter/higher contrast).
    
    Returns:
        Augmented PIL Image.
    """
    img = ImageEnhance.Brightness(img).enhance(factor)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def positive_intensity(img, factor, **__):
    """Brightening-only intensity adjustment (factor >= 1.0)."""
    img = ImageEnhance.Brightness(img).enhance(factor)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def negative_intensity(img, factor, **__):
    """Darkening-only intensity adjustment (factor <= 1.0)."""
    img = ImageEnhance.Brightness(img).enhance(factor)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def desaturate(img, factor, **_):
    factor = min(1., max(0., 1. - factor))
    # enhance factor 0 = grayscale, 1.0 = no-change
    return ImageEnhance.Color(img).enhance(factor)


def salt_and_pepper(img, amount, **__):
    """Apply salt and pepper noise to an image.
    
    Randomly sets pixels to either black (pepper) or white (salt).
    
    Args:
        img: PIL Image to augment.
        amount: Proportion of pixels to affect (0.0 to 1.0).
                Higher values = more noise.
    
    Returns:
        Augmented PIL Image with salt and pepper noise.
    """
    import numpy as np
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Get image shape
    if len(img_array.shape) == 2:
        # Grayscale
        h, w = img_array.shape
        c = 1
    else:
        h, w, c = img_array.shape
    
    # Calculate number of pixels to affect
    num_pixels = int(amount * h * w)
    
    # Salt (white pixels)
    num_salt = num_pixels // 2
    salt_coords = (
        np.random.randint(0, h, num_salt),
        np.random.randint(0, w, num_salt)
    )
    
    # Pepper (black pixels)
    num_pepper = num_pixels - num_salt
    pepper_coords = (
        np.random.randint(0, h, num_pepper),
        np.random.randint(0, w, num_pepper)
    )
    
    # Apply noise
    img_array = img_array.copy()
    if len(img_array.shape) == 2:
        img_array[salt_coords] = 255
        img_array[pepper_coords] = 0
    else:
        img_array[salt_coords[0], salt_coords[1], :] = 255
        img_array[pepper_coords[0], pepper_coords[1], :] = 0
    
    return Image.fromarray(img_array)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _LEVEL_DENOM) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * .9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return level,


def _enhance_deterministic_level_to_arg(level, _hparams, direction='increase'):
    """Deterministic version of _enhance_increasing_level_to_arg (no random negation).
    
    This is useful for visualization and evaluation where reproducible results are needed.
    
    Args:
        level: Magnitude level (0-10 scale).
        _hparams: Hyperparameters (unused).
        direction: 'increase' for brightening (factor > 1), 'decrease' for darkening (factor < 1).
    
    Returns:
        Enhancement factor tuple.
    """
    # Map level [0, 10] to factor change [0, 0.9]
    level = (level / _LEVEL_DENOM) * .9
    if direction == 'increase':
        # Brightening: factor in [1.0, 1.9]
        factor = 1.0 + level
    else:
        # Darkening: factor in [0.1, 1.0]
        factor = max(0.1, 1.0 - level)
    return factor,


def _positive_intensity_level_to_arg(level, _hparams):
    """Brightening: factor in [1.0, 1.9]. No random negation."""
    level = (level / _LEVEL_DENOM) * .9
    return 1.0 + level,


def _negative_intensity_level_to_arg(level, _hparams):
    """Darkening: factor in [0.1, 1.0]. No random negation."""
    level = (level / _LEVEL_DENOM) * .9
    return max(0.1, 1.0 - level),


def _saturation_v2_level_to_arg(level, _hparams):
    """Extended saturation for training. Range [1.0, 5.0] with exponential scaling."""
    normalized = level / _LEVEL_DENOM
    factor = 1.0 + 4.0 * (normalized ** 1.3)
    return factor,


def _sharpness_v2_level_to_arg(level, _hparams):
    """Extended sharpness for training. Range [1.0, 10.0] with exponential scaling."""
    normalized = level / _LEVEL_DENOM
    factor = 1.0 + 9.0 * (normalized ** 1.3)
    return factor,


def _minmax_level_to_arg(level, _hparams, min_val=0., max_val=1.0, clamp=True):
    level = (level / _LEVEL_DENOM)
    level = min_val + (max_val - min_val) * level
    if clamp:
        level = max(min_val, min(max_val, level))
    return level,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _LEVEL_DENOM) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 1], 'keep 4 down to 1 MSB of original image',
    # intensity/severity of augmentation increases with level
    # Note: minimum 1 bit to avoid completely black images
    bits = 4 - _posterize_level_to_arg(level, hparams)[0]
    return max(1, bits),


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return min(256, int((level / _LEVEL_DENOM) * 256)),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return min(128, int((level / _LEVEL_DENOM) * 110)),


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
    'Desaturate': partial(_minmax_level_to_arg, min_val=0.5, max_val=1.0),
    'GaussianBlur': partial(_minmax_level_to_arg, min_val=0.1, max_val=2.0),
    'GaussianBlurRand': _minmax_level_to_arg,
    # V2 transforms for augmentation prediction task
    'IntensityIncreasing': _enhance_increasing_level_to_arg,  # Combined Brightness + Contrast (legacy)
    'SaturationIncreasing': _enhance_increasing_level_to_arg,  # Same as ColorIncreasing (renamed for semantics)
    'GaussianBlurIncreasing': partial(_minmax_level_to_arg, min_val=0.1, max_val=2.0),  # Blur with increasing severity
    'SaltAndPepperIncreasing': partial(_minmax_level_to_arg, min_val=0.0, max_val=0.3),  # Salt and pepper noise
    # V2.1 transforms: split Intensity + extended Saturation/Sharpness
    'PositiveIntensity': _positive_intensity_level_to_arg,
    'NegativeIntensity': _negative_intensity_level_to_arg,
    'SaturationV2': _saturation_v2_level_to_arg,
    'SharpnessV2': _sharpness_v2_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
    'Desaturate': desaturate,
    'GaussianBlur': gaussian_blur,
    'GaussianBlurRand': gaussian_blur_rand,
    # V2 transforms for augmentation prediction task
    'IntensityIncreasing': intensity,  # Combined Brightness + Contrast (legacy)
    'SaturationIncreasing': color,  # Same as ColorIncreasing (renamed for semantics)
    'GaussianBlurIncreasing': gaussian_blur,  # Blur with increasing severity
    'SaltAndPepperIncreasing': salt_and_pepper,  # Salt and pepper noise
    # V2.1 transforms: split Intensity + extended Saturation/Sharpness
    'PositiveIntensity': positive_intensity,
    'NegativeIntensity': negative_intensity,
    'SaturationV2': color,
    'SharpnessV2': sharpness,
}

GEOMETRIC_OPS = {
    'Rotate', 'ShearX', 'ShearY',
    'TranslateX', 'TranslateY', 'TranslateXRel', 'TranslateYRel',
}

COLOR_OPS = set(NAME_TO_OP.keys()) - GEOMETRIC_OPS


class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)
        self.magnitude_max = self.hparams.get('magnitude_max', None)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float('inf'):
                # inf == uniform sampling
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        fs = self.__class__.__name__ + f'(name={self.name}, p={self.prob}'
        fs += f', m={self.magnitude}, mstd={self.magnitude_std}'
        if self.magnitude_max is not None:
            fs += f', mmax={self.magnitude_max}'
        fs += ')'
        return fs


def auto_augment_policy_v0(hparams):
    # ImageNet v0 policy from TPU EfficientNet impl, cannot find a paper reference.
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],  # This results in black image with Tpu posterize
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_v0r(hparams):
    # ImageNet v0 policy from TPU EfficientNet impl, with variation of Posterize used
    # in Google research implementation (number of bits discarded increases with magnitude)
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('PosterizeIncreasing', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_original(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501
    policy = [
        [('PosterizeOriginal', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeOriginal', 0.6, 7), ('PosterizeOriginal', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeOriginal', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeOriginal', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_originalr(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501 with research posterize variation
    policy = [
        [('PosterizeIncreasing', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeIncreasing', 0.6, 7), ('PosterizeIncreasing', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeIncreasing', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_3a(hparams):
    policy = [
        [('Solarize', 1.0, 5)],  # 128 solarize threshold @ 5 magnitude
        [('Desaturate', 1.0, 10)],  # grayscale at 10 magnitude
        [('GaussianBlurRand', 1.0, 10)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name='v0', hparams=None):
    hparams = hparams or _HPARAMS_DEFAULT
    if name == 'original':
        return auto_augment_policy_original(hparams)
    if name == 'originalr':
        return auto_augment_policy_originalr(hparams)
    if name == 'v0':
        return auto_augment_policy_v0(hparams)
    if name == 'v0r':
        return auto_augment_policy_v0r(hparams)
    if name == '3a':
        return auto_augment_policy_3a(hparams)
    assert False, f'Unknown AA policy {name}'


class AutoAugment:

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img

    def __repr__(self):
        fs = self.__class__.__name__ + '(policy='
        for p in self.policy:
            fs += '\n\t['
            fs += ', '.join([str(op) for op in p])
            fs += ']'
        fs += ')'
        return fs


def auto_augment_transform(config_str: str, hparams: Optional[Dict] = None):
    """ Create a AutoAugment transform

    Args:
        config_str: String defining configuration of auto augmentation. Consists of multiple sections separated by
            dashes ('-').
            The first section defines the AutoAugment policy (one of 'v0', 'v0r', 'original', 'originalr').
            While the remaining sections define other arguments
                * 'mstd' -  float std deviation of magnitude noise applied
        hparams: Other hparams (kwargs) for the AutoAugmentation scheme

    Returns:
         A PyTorch compatible Transform

    Examples::

        'original-mstd0.5' results in AutoAugment with original policy, magnitude_std 0.5
    """
    config = config_str.split('-')
    policy_name = config[0]
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        else:
            assert False, 'Unknown AutoAugment config section'
    aa_policy = auto_augment_policy(policy_name, hparams=hparams)
    return AutoAugment(aa_policy)


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]


_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]


_RAND_3A = [
    'SolarizeIncreasing',
    'Desaturate',
    'GaussianBlur',
]


_RAND_WEIGHTED_3A = {
    'SolarizeIncreasing': 6,
    'Desaturate': 6,
    'GaussianBlur': 6,
    'Rotate': 3,
    'ShearX': 2,
    'ShearY': 2,
    'PosterizeIncreasing': 1,
    'AutoContrast': 1,
    'ColorIncreasing': 1,
    'SharpnessIncreasing': 1,
    'ContrastIncreasing': 1,
    'BrightnessIncreasing': 1,
    'Equalize': 1,
    'Invert': 1,
}


# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_WEIGHTED_0 = {
    'Rotate': 3,
    'ShearX': 2,
    'ShearY': 2,
    'TranslateXRel': 1,
    'TranslateYRel': 1,
    'ColorIncreasing': .25,
    'SharpnessIncreasing': 0.25,
    'AutoContrast': 0.25,
    'SolarizeIncreasing': .05,
    'SolarizeAdd': .05,
    'ContrastIncreasing': .05,
    'BrightnessIncreasing': .05,
    'Equalize': .05,
    'PosterizeIncreasing': 0.05,
    'Invert': 0.05,
}


def _get_weighted_transforms(transforms: Dict):
    transforms, probs = list(zip(*transforms.items()))
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return transforms, probs


def rand_augment_choices(name: str, increasing=True):
    if name == 'weights':
        return _RAND_WEIGHTED_0
    if name == '3aw':
        return _RAND_WEIGHTED_3A
    if name == '3a':
        return _RAND_3A
    return _RAND_INCREASING_TRANSFORMS if increasing else _RAND_TRANSFORMS


def rand_augment_ops(
        magnitude: Union[int, float] = 10,
        prob: float = 0.5,
        hparams: Optional[Dict] = None,
        transforms: Optional[Union[Dict, List]] = None,
):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(
        name, prob=prob, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


def rand_augment_transform(
        config_str: str,
        hparams: Optional[Dict] = None,
        transforms: Optional[Union[str, Dict, List]] = None,
):
    """ Create a RandAugment transform

    Args:
        config_str (str): String defining configuration of random augmentation. Consists of multiple sections separated
            by dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand').
            The remaining sections, not order specific determine
                * 'm' - integer magnitude of rand augment
                * 'n' - integer num layers (number of transform ops selected per image)
                * 'p' - float probability of applying each layer (default 0.5)
                * 'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100)
                * 'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10)
                * 'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
                * 't' - str name of transform set to use
        hparams (dict): Other hparams (kwargs) for the RandAugmentation scheme

    Returns:
         A PyTorch compatible Transform

    Examples::

        'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5

        'rand-mstd1-tweights' results in mag std 1.0, weighted transforms, default mag of 10 and num_layers 2

    """
    magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    increasing = False
    prob = 0.5
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        if c.startswith('t'):
            # NOTE old 'w' key was removed, 'w0' is not equivalent to 'tweights'
            val = str(c[1:])
            if transforms is None:
                transforms = val
        else:
            # numeric options
            cs = re.split(r'(\d.*)', c)
            if len(cs) < 2:
                continue
            key, val = cs[:2]
            if key == 'mstd':
                # noise param / randomization of magnitude values
                mstd = float(val)
                if mstd > 100:
                    # use uniform sampling in 0 to magnitude if mstd is > 100
                    mstd = float('inf')
                hparams.setdefault('magnitude_std', mstd)
            elif key == 'mmax':
                # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
                hparams.setdefault('magnitude_max', int(val))
            elif key == 'inc':
                if bool(val):
                    increasing = True
            elif key == 'm':
                magnitude = int(val)
            elif key == 'n':
                num_layers = int(val)
            elif key == 'p':
                prob = float(val)
            else:
                assert False, 'Unknown RandAugment config section'

    if isinstance(transforms, str):
        transforms = rand_augment_choices(transforms, increasing=increasing)
    elif transforms is None:
        transforms = _RAND_INCREASING_TRANSFORMS if increasing else _RAND_TRANSFORMS

    choice_weights = None
    if isinstance(transforms, Dict):
        transforms, choice_weights = _get_weighted_transforms(transforms)

    ra_ops = rand_augment_ops(magnitude=magnitude, prob=prob, hparams=hparams, transforms=transforms)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


_AUGMIX_TRANSFORMS = [
    'AutoContrast',
    'ColorIncreasing',  # not in paper
    'ContrastIncreasing',  # not in paper
    'BrightnessIncreasing',  # not in paper
    'SharpnessIncreasing',  # not in paper
    'Equalize',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]


def augmix_ops(
        magnitude: Union[int, float] = 10,
        hparams: Optional[Dict] = None,
        transforms: Optional[Union[str, Dict, List]] = None,
):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _AUGMIX_TRANSFORMS
    return [AugmentOp(
        name,
        prob=1.0,
        magnitude=magnitude,
        hparams=hparams
    ) for name in transforms]


class AugMixAugment:
    """ AugMix Transform
    Adapted and improved from impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """
    def __init__(self, ops, alpha=1., width=3, depth=-1, blended=False):
        self.ops = ops
        self.alpha = alpha
        self.width = width
        self.depth = depth
        self.blended = blended  # blended mode is faster but not well tested

    def _calc_blended_weights(self, ws, m):
        ws = ws * m
        cump = 1.
        rws = []
        for w in ws[::-1]:
            alpha = w / cump
            cump *= (1 - alpha)
            rws.append(alpha)
        return np.array(rws[::-1], dtype=np.float32)

    def _apply_blended(self, img, mixing_weights, m):
        # This is my first crack and implementing a slightly faster mixed augmentation. Instead
        # of accumulating the mix for each chain in a Numpy array and then blending with original,
        # it recomputes the blending coefficients and applies one PIL image blend per chain.
        # TODO the results appear in the right ballpark but they differ by more than rounding.
        img_orig = img.copy()
        ws = self._calc_blended_weights(mixing_weights, m)
        for w in ws:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img_orig  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            img = Image.blend(img, img_aug, w)
        return img

    def _apply_basic(self, img, mixing_weights, m):
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        img_shape = img.size[0], img.size[1], len(img.getbands())
        mixed = np.zeros(img_shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * np.asarray(img_aug, dtype=np.float32)
        np.clip(mixed, 0, 255., out=mixed)
        mixed = Image.fromarray(mixed.astype(np.uint8))
        return Image.blend(img, mixed, m)

    def __call__(self, img):
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        if self.blended:
            mixed = self._apply_blended(img, mixing_weights, m)
        else:
            mixed = self._apply_basic(img, mixing_weights, m)
        return mixed

    def __repr__(self):
        fs = self.__class__.__name__ + f'(alpha={self.alpha}, width={self.width}, depth={self.depth}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


def augment_and_mix_transform(config_str: str, hparams: Optional[Dict] = None):
    """ Create AugMix PyTorch transform

    Args:
        config_str (str): String defining configuration of random augmentation. Consists of multiple sections separated
            by dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand').
            The remaining sections, not order specific determine
                'm' - integer magnitude (severity) of augmentation mix (default: 3)
                'w' - integer width of augmentation chain (default: 3)
                'd' - integer depth of augmentation chain (-1 is random [1, 3], default: -1)
                'b' - integer (bool), blend each branch of chain into end result without a final blend, less CPU (default: 0)
                'mstd' -  float std deviation of magnitude noise applied (default: 0)
            Ex 'augmix-m5-w4-d2' results in AugMix with severity 5, chain width 4, chain depth 2

        hparams: Other hparams (kwargs) for the Augmentation transforms

    Returns:
         A PyTorch compatible Transform
    """
    magnitude = 3
    width = 3
    depth = -1
    alpha = 1.
    blended = False
    config = config_str.split('-')
    assert config[0] == 'augmix'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'w':
            width = int(val)
        elif key == 'd':
            depth = int(val)
        elif key == 'a':
            alpha = float(val)
        elif key == 'b':
            blended = bool(val)
        else:
            assert False, 'Unknown AugMix config section'
    hparams.setdefault('magnitude_std', float('inf'))  # default to uniform sampling (if not set via mstd arg)
    ops = augmix_ops(magnitude=magnitude, hparams=hparams)
    return AugMixAugment(ops, alpha=alpha, width=width, depth=depth, blended=blended)


# =============================================================================
# New AugMix SL (Severity Level) Controllable Policy
# - Only includes transforms with controllable severity levels
# - Excludes geometric transforms (Rotate, ShearX, ShearY, TranslateX, TranslateY)
# - Excludes transforms without severity control (AutoContrast, Equalize)
# =============================================================================

_AUGMIX_SL_TRANSFORMS = [
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
]

# Index mapping for transforms (for ground truth labels)
AUGMIX_SL_TRANSFORM_TO_IDX = {name: idx for idx, name in enumerate(_AUGMIX_SL_TRANSFORMS)}
AUGMIX_SL_IDX_TO_TRANSFORM = {idx: name for idx, name in enumerate(_AUGMIX_SL_TRANSFORMS)}
AUGMIX_SL_NUM_TRANSFORMS = len(_AUGMIX_SL_TRANSFORMS)


# =============================================================================
# AugMix SL V2 Policy
# - Semantically reorganized transforms:
#   - PositiveIntensity: Brightening (factor > 1.0)
#   - NegativeIntensity: Darkening (factor < 1.0)
#   - SaturationV2: Color/Saturation with extended range [1.0, 5.0]
#   - SharpnessV2: Edge sharpness with extended range [1.0, 10.0]
#   - GaussianBlurIncreasing: Blur effect
#   - PosterizeIncreasing: Color quantization
#   - SolarizeIncreasing: Inversion based on threshold
#   - SaltAndPepperIncreasing: Salt and pepper noise
# =============================================================================

_AUGMIX_SL_TRANSFORMS_V2 = [
    'PositiveIntensity',        # Brightening (밝아짐)
    'NegativeIntensity',        # Darkening (어두워짐)
    'SaturationV2',             # Color/Saturation 확장 범위 (채도)
    'SharpnessV2',              # 선명도 확장 범위
    'GaussianBlurIncreasing',   # 블러
    'PosterizeIncreasing',      # 포스터화
    'SolarizeIncreasing',       # 솔라라이즈
    'SaltAndPepperIncreasing',  # Salt and Pepper 노이즈
]

# Index mapping for V2 transforms
AUGMIX_SL_V2_TRANSFORM_TO_IDX = {name: idx for idx, name in enumerate(_AUGMIX_SL_TRANSFORMS_V2)}
AUGMIX_SL_V2_IDX_TO_TRANSFORM = {idx: name for idx, name in enumerate(_AUGMIX_SL_TRANSFORMS_V2)}
AUGMIX_SL_V2_NUM_TRANSFORMS = len(_AUGMIX_SL_TRANSFORMS_V2)


class AugMixSLOp:
    """Augmentation operation with normalized severity level (0-1).
    
    Unlike the standard AugmentOp, this class:
    - Always applies the operation (prob=1.0)
    - Uses normalized magnitude (0-1 range) that gets scaled to [0, 10]
    - Returns both the augmented image and the applied magnitude
    """

    def __init__(self, name: str, hparams: Optional[Dict] = None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

    def __call__(self, img, normalized_magnitude: float):
        """Apply augmentation with specified normalized magnitude.
        
        Args:
            img: PIL Image to augment.
            normalized_magnitude: Magnitude in [0, 1] range.
        
        Returns:
            Augmented PIL Image.
        """
        # Scale normalized magnitude to [0, 10] for the level functions
        magnitude = normalized_magnitude * _LEVEL_DENOM
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'


def augmix_sl_ops(hparams: Optional[Dict] = None) -> List[AugMixSLOp]:
    """Create list of SL-controllable augmentation operations.
    
    Args:
        hparams: Optional hyperparameters for augmentation.
    
    Returns:
        List of AugMixSLOp instances.
    """
    hparams = hparams or _HPARAMS_DEFAULT
    return [AugMixSLOp(name, hparams=hparams) for name in _AUGMIX_SL_TRANSFORMS]


def augmix_sl_ops_v2(hparams: Optional[Dict] = None) -> List[AugMixSLOp]:
    """Create list of SL-controllable augmentation operations (V2 policy).
    
    V2 policy with 8 deterministic-direction transforms:
    - PositiveIntensity: Brightening (factor > 1.0)
    - NegativeIntensity: Darkening (factor < 1.0)
    - SaturationV2: Extended saturation range [1.0, 5.0]
    - SharpnessV2: Extended sharpness range [1.0, 10.0]
    - GaussianBlurIncreasing: Blur effect
    - PosterizeIncreasing: Color quantization
    - SolarizeIncreasing: Inversion based on threshold
    - SaltAndPepperIncreasing: Salt and pepper noise
    
    Args:
        hparams: Optional hyperparameters for augmentation.
    
    Returns:
        List of AugMixSLOp instances.
    """
    hparams = hparams or _HPARAMS_DEFAULT
    return [AugMixSLOp(name, hparams=hparams) for name in _AUGMIX_SL_TRANSFORMS_V2]


# =============================================================================
# Deterministic Augmentation Ops for Visualization
# =============================================================================

# Deterministic level-to-arg mappings (no random negation)
# Extended ranges for better visualization

def _sharpness_extended_level_to_arg(level, _hparams):
    """Extended sharpness range for more visible effect.
    
    PIL ImageEnhance.Sharpness:
    - 0.0 = blurred
    - 1.0 = original
    - 2.0+ = sharpened
    
    Range: [1.0, 20.0] for very visible sharpening effect.
    Uses exponential scaling for more visible difference across SL levels.
    """
    # Map level [0, 10] to factor [1.0, 20.0] with exponential scaling
    normalized = level / _LEVEL_DENOM  # 0 to 1
    # Exponential scaling: factor = 1.0 + 19.0 * (normalized^1.5)
    # This gives: SL=0.1 -> 1.6, SL=0.5 -> 7.7, SL=1.0 -> 20.0
    factor = 1.0 + 19.0 * (normalized ** 1.5)
    return factor,


def _saturation_extended_level_to_arg(level, _hparams):
    """Extended saturation range for more visible effect.
    
    PIL ImageEnhance.Color:
    - 0.0 = grayscale
    - 1.0 = original
    - 2.0+ = more saturated
    
    Range: [1.0, 7.0] for very visible saturation effect.
    Uses exponential scaling for more visible difference across SL levels.
    """
    # Map level [0, 10] to factor [1.0, 7.0] with exponential scaling
    normalized = level / _LEVEL_DENOM  # 0 to 1
    # Exponential scaling: factor = 1.0 + 6.0 * (normalized^1.5)
    # This gives: SL=0.1 -> 1.2, SL=0.5 -> 3.1, SL=1.0 -> 7.0
    factor = 1.0 + 6.0 * (normalized ** 1.5)
    return factor,


def _gaussian_blur_extended_level_to_arg(level, _hparams):
    """Extended gaussian blur range for more visible effect.
    
    Range: [0.1, 8.0] for visible blur effect.
    """
    # Map level [0, 10] to radius [0.1, 8.0]
    min_val, max_val = 0.1, 8.0
    radius = min_val + (level / _LEVEL_DENOM) * (max_val - min_val)
    return radius,


def _salt_and_pepper_extended_level_to_arg(level, _hparams):
    """Extended salt and pepper noise range for visible effect.
    
    Range: [0.0, 0.9] - proportion of pixels affected.
    0.0 = no noise, 0.9 = 90% of pixels are salt or pepper.
    Uses exponential scaling for more visible difference at lower SLs.
    """
    # Map level [0, 10] to amount [0.0, 0.9] with exponential scaling
    # This makes the difference more visible across all severity levels
    normalized = level / _LEVEL_DENOM  # 0 to 1
    # Exponential scaling: amount = 0.9 * (normalized^1.5)
    # This gives: SL=0.1 -> 0.028, SL=0.5 -> 0.32, SL=1.0 -> 0.9
    amount = 0.9 * (normalized ** 1.5)
    return amount,


_DETERMINISTIC_LEVEL_TO_ARG = {
    # Legacy (kept for backward compatibility)
    'IntensityIncreasing': partial(_enhance_deterministic_level_to_arg, direction='increase'),
    'SaturationIncreasing': _saturation_extended_level_to_arg,
    'SharpnessIncreasing': _sharpness_extended_level_to_arg,
    # V2 transforms
    'PositiveIntensity': _positive_intensity_level_to_arg,
    'NegativeIntensity': _negative_intensity_level_to_arg,
    'SaturationV2': _saturation_extended_level_to_arg,        # Extended range [1.0, 7.0]
    'SharpnessV2': _sharpness_extended_level_to_arg,          # Extended range [1.0, 20.0]
    'GaussianBlurIncreasing': _gaussian_blur_extended_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SaltAndPepperIncreasing': _salt_and_pepper_extended_level_to_arg,
}


class AugMixSLOpDeterministic:
    """Deterministic augmentation operation for visualization.
    
    Unlike AugMixSLOp, this class uses deterministic level-to-arg functions
    without random negation, ensuring reproducible results for visualization.
    """

    def __init__(self, name: str, hparams: Optional[Dict] = None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        # Use deterministic level_fn if available, otherwise fall back to original
        if name in _DETERMINISTIC_LEVEL_TO_ARG:
            self.level_fn = _DETERMINISTIC_LEVEL_TO_ARG[name]
        else:
            self.level_fn = LEVEL_TO_ARG[name]
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

    def __call__(self, img, normalized_magnitude: float):
        """Apply augmentation with specified normalized magnitude (deterministic).
        
        Args:
            img: PIL Image to augment.
            normalized_magnitude: Magnitude in [0, 1] range.
        
        Returns:
            Augmented PIL Image.
        """
        # Scale normalized magnitude to [0, 10] for the level functions
        magnitude = normalized_magnitude * _LEVEL_DENOM
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'


def augmix_sl_ops_v2_deterministic(hparams: Optional[Dict] = None) -> List[AugMixSLOpDeterministic]:
    """Create list of deterministic SL-controllable augmentation operations (V2 policy).
    
    This is the deterministic version for visualization purposes.
    Unlike augmix_sl_ops_v2, this version does NOT use random negation,
    so the results are reproducible and consistent.
    
    Args:
        hparams: Optional hyperparameters for augmentation.
    
    Returns:
        List of AugMixSLOpDeterministic instances.
    """
    hparams = hparams or _HPARAMS_DEFAULT
    return [AugMixSLOpDeterministic(name, hparams=hparams) for name in _AUGMIX_SL_TRANSFORMS_V2]


# Helper functions for both V1 and V2
def get_augmix_sl_num_transforms(version: int = 1) -> int:
    """Get number of transforms for specified version."""
    if version == 2:
        return AUGMIX_SL_V2_NUM_TRANSFORMS
    return AUGMIX_SL_NUM_TRANSFORMS


def get_augmix_sl_transform_names(version: int = 1) -> List[str]:
    """Get list of transform names for specified version."""
    if version == 2:
        return list(_AUGMIX_SL_TRANSFORMS_V2)
    return list(_AUGMIX_SL_TRANSFORMS)


def get_augmix_sl_transform_to_idx(version: int = 1) -> Dict[str, int]:
    """Get transform name to index mapping for specified version."""
    if version == 2:
        return AUGMIX_SL_V2_TRANSFORM_TO_IDX
    return AUGMIX_SL_TRANSFORM_TO_IDX


def get_augmix_sl_idx_to_transform(version: int = 1) -> Dict[int, str]:
    """Get index to transform name mapping for specified version."""
    if version == 2:
        return AUGMIX_SL_V2_IDX_TO_TRANSFORM
    return AUGMIX_SL_IDX_TO_TRANSFORM


class AugMixSLAugment:
    """AugMix Transform with Severity Level (SL) labels for augmentation prediction task.

    Severity shares always sum to 1.0 (probability distribution):
    - depth=1 → single transform gets share=1.0 (applied at max_sl).
    - depth=K → shares drawn from Dirichlet(1,...,1), each applied at share*max_sl.

    Ground truth format:
        - Array of shape ``[num_transforms]``:
          - 0.0 if the transform was NOT applied.
          - share (0, 1] if applied.  All non-zero entries sum to 1.0.

    Args:
        ops: List of AugMixSLOp instances.
        max_depth: Maximum number of transforms to apply (1-3 randomly chosen).
        min_sl: Minimum severity level (kept for API compat, not used in simplex sampling).
        max_sl: Maximum severity level (default: 1.0). Physical severity = share * max_sl.
        normalize_labels: Ignored (labels always sum to 1.0). Kept for API compatibility.
    """

    def __init__(
        self,
        ops: List[AugMixSLOp],
        max_depth: int = 3,
        min_sl: float = 0.1,
        max_sl: float = 1.0,
        normalize_labels: bool = False,
    ):
        self.ops = ops
        self.num_ops = len(ops)
        self.max_depth = max_depth
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.normalize_labels = normalize_labels  # kept for API compat

        # Create name to index mapping
        self.op_to_idx = {op.name: idx for idx, op in enumerate(self.ops)}

    def __call__(self, img):
        """Apply random augmentations and return image with labels.

        Severity levels are sampled so that they **always sum to 1.0**,
        forming a valid probability distribution over transforms.

        - depth=1 → the single selected transform gets SL=1.0.
        - depth=K>1 → SL shares are drawn from a symmetric Dirichlet(1)
          (i.e. uniform on the simplex), giving K values that sum to 1.0.

        The actual augmentation is applied at ``share * max_sl`` so that
        the physical severity stays within [0, max_sl].

        Args:
            img: PIL Image to augment.

        Returns:
            Tuple of (augmented_image, aug_labels):
                - augmented_image: PIL Image after augmentation.
                - aug_labels: numpy array ``[num_transforms]`` — a probability
                  distribution (sums to 1.0).  0 for unapplied transforms.
        """
        # Initialize labels (all zeros = no transform applied)
        aug_labels = np.zeros(self.num_ops, dtype=np.float32)

        # Randomly choose number of transforms to apply (1 to max_depth)
        num_transforms = np.random.randint(1, self.max_depth + 1)

        # Randomly select which transforms to apply (without replacement)
        selected_indices = np.random.choice(self.num_ops, size=num_transforms, replace=False)

        # Sample severity shares that sum to 1.0
        if num_transforms == 1:
            shares = np.array([1.0])
        else:
            # Dirichlet(1,...,1) = uniform on the simplex → shares sum to 1
            shares = np.random.dirichlet(np.ones(num_transforms))

        # Apply each selected transform
        for i, idx in enumerate(selected_indices):
            op = self.ops[idx]
            share = float(shares[i])
            # Actual severity for the transform: share * max_sl
            sl = share * self.max_sl
            img = op(img, sl)
            # Label is the share itself (probability, sums to 1.0)
            aug_labels[idx] = share

        return img, aug_labels

    def __repr__(self):
        fs = self.__class__.__name__ + f'(max_depth={self.max_depth}, '
        fs += f'min_sl={self.min_sl}, max_sl={self.max_sl}, '
        fs += f'normalize_labels={self.normalize_labels}, ops=['
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += '\n])'
        return fs


class AugMixSLAugmentFixed:
    """Fixed augmentation for validation - applies predetermined transforms.
    
    This class is used during validation to apply a fixed set of augmentations
    with fixed severity levels, allowing consistent evaluation.
    
    Args:
        ops: List of AugMixSLOp instances.
        fixed_transforms: List of (transform_index, severity_level) tuples.
        max_sl: Maximum severity level for scaling (SL is scaled by max_sl so max_sl maps to 1.0).
    """

    def __init__(
        self,
        ops: List[AugMixSLOp],
        fixed_transforms: List[Tuple[int, float]],
        max_sl: float = 1.0,
    ):
        self.ops = ops
        self.num_ops = len(ops)
        self.fixed_transforms = fixed_transforms
        self.max_sl = max_sl

    def __call__(self, img):
        """Apply fixed augmentations and return image with labels.
        
        Args:
            img: PIL Image to augment.
        
        Returns:
            Tuple of (augmented_image, aug_labels).
            SL values are scaled by max_sl so that max_sl maps to 1.0.
        """
        aug_labels = np.zeros(self.num_ops, dtype=np.float32)
        
        for idx, sl in self.fixed_transforms:
            op = self.ops[idx]
            img = op(img, sl)
            # Scale SL by max_sl so that max_sl maps to 1.0
            scaled_sl = sl / self.max_sl
            aug_labels[idx] = scaled_sl
        
        return img, aug_labels

    def __repr__(self):
        return f'{self.__class__.__name__}(max_sl={self.max_sl}, fixed_transforms={self.fixed_transforms})'


def create_augmix_sl_transform(
    max_depth: int = 3,
    min_sl: float = 0.1,
    max_sl: float = 1.0,
    hparams: Optional[Dict] = None,
    version: int = 1,
    normalize_labels: bool = False,
) -> AugMixSLAugment:
    """Create AugMixSL transform for training.
    
    Args:
        max_depth: Maximum number of transforms to apply per image.
        min_sl: Minimum severity level.
        max_sl: Maximum severity level.
        hparams: Optional hyperparameters for augmentation.
        version: Policy version (1 for original, 2 for V2 with PositiveIntensity etc.)
        normalize_labels: If True, normalize aug_labels to sum to 1 (probability distribution).
    
    Returns:
        AugMixSLAugment instance.
    """
    if version == 2:
        ops = augmix_sl_ops_v2(hparams=hparams)
    else:
        ops = augmix_sl_ops(hparams=hparams)
    return AugMixSLAugment(
        ops, 
        max_depth=max_depth, 
        min_sl=min_sl, 
        max_sl=max_sl,
        normalize_labels=normalize_labels,
    )


def create_augmix_sl_validation_transforms(
    num_groups: int,
    max_depth: int = 3,
    hparams: Optional[Dict] = None,
    version: int = 1,
    min_sl: float = 0.5,
    max_sl: float = 1.0,
) -> List[AugMixSLAugmentFixed]:
    """Create fixed augmentation transforms for validation.
    
    Creates multiple fixed transform combinations to ensure all transforms
    are covered during validation. Each group gets a different combination.
    Uses stronger severity levels for clearer validation signals.
    
    Args:
        num_groups: Number of different validation groups.
        max_depth: Number of transforms per group.
        hparams: Optional hyperparameters for augmentation.
        version: Policy version (1 for original, 2 for V2 with PositiveIntensity etc.)
        min_sl: Minimum severity level for validation (default: 0.5).
        max_sl: Maximum severity level for validation (default: 1.0).
    
    Returns:
        List of AugMixSLAugmentFixed instances, one per group.
    """
    if version == 2:
        ops = augmix_sl_ops_v2(hparams=hparams)
    else:
        ops = augmix_sl_ops(hparams=hparams)
    num_ops = len(ops)
    
    # Create balanced assignments - ensure each transform appears roughly equally
    validation_transforms = []
    
    # Fixed severity levels for validation - use stronger values for clearer signal
    # Spread between min_sl and max_sl
    fixed_sls = [min_sl, (min_sl + max_sl) / 2, max_sl]
    
    # Generate all possible combinations of max_depth transforms from num_ops
    # Use itertools.combinations for better coverage
    from itertools import combinations
    all_combos = list(combinations(range(num_ops), max_depth))
    
    for group_idx in range(num_groups):
        # Cycle through all combinations
        combo = all_combos[group_idx % len(all_combos)]
        fixed_pairs = []
        for i, transform_idx in enumerate(combo):
            sl = fixed_sls[i % len(fixed_sls)]
            fixed_pairs.append((transform_idx, sl))
        
        validation_transforms.append(AugMixSLAugmentFixed(ops, fixed_pairs, max_sl=max_sl))
    
    return validation_transforms
