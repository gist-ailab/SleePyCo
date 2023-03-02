import torch
import random
import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift


class TwoTransform:
    
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class Compose:

    def __init__(self, transforms, mode='full'):
        self.transforms = transforms
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'random':
            index = random.randint(0, len(self.transforms) - 1)
            x = self.transforms[index](x)
        elif self.mode == 'full':
            for t in self.transforms:
                x = t(x)
        elif self.mode == 'shuffle':
            transforms = np.random.choice(self.transforms, len(self.transforms), replace=False)
            for t in transforms:
                x = t(x)
        else:
            raise NotImplementedError
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomAmplitudeScale:

    def __init__(self, range=(0.5, 2.0), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            scale = random.uniform(self.range[0], self.range[1])
            return x * scale
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomDCShift:
    
    def __init__(self, range=(-10.0, 10.0), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            shift = random.uniform(self.range[0], self.range[1])
            return x + shift
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTimeShift:

    def __init__(self, range=(-300, 300), mode='constant', cval=0.0, p=0.5):
        self.range = range
        self.mode = mode
        self.cval = cval
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            t_shift = random.randint(self.range[0], self.range[1])
            if len(x.shape) == 2:
                x = x[0]
            x = shift(input=x, shift=t_shift, mode=self.mode, cval=self.cval)
            x = np.expand_dims(x, axis=0)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomZeroMasking:
    
    def __init__(self, range=(0, 300), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            mask_len = random.randint(self.range[0], self.range[1])
            random_pos = random.randint(0, x.shape[1] - mask_len)
            mask = np.concatenate([np.ones((1, random_pos)), np.zeros((1, mask_len)), np.ones((1, x.shape[1] - mask_len - random_pos))], axis=1)
            return x * mask
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomAdditiveGaussianNoise:
    
    def __init__(self, range=(0.0, 0.2), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            sigma = random.uniform(self.range[0], self.range[1])
            return x + np.random.normal(0, sigma, x.shape)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomBandStopFilter:

    def __init__(self, range=(0.5, 30.0), band_width=2.0, sampling_rate=100.0, p=0.5):
        self.range = range
        self.band_width = band_width
        self.sampling_rate = sampling_rate
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            low_freq = random.uniform(self.range[0], self.range[1])
            center_freq = low_freq + self.band_width / 2.0
            b, a = signal.iirnotch(center_freq, center_freq / self.band_width, fs=self.sampling_rate)
            x = signal.lfilter(b, a, x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
