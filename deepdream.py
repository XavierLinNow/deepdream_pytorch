import numpy as np
import torch
from util import showtensor
import scipy.ndimage as nd
from torch.autograd import Variable

def objective_L2(dst, guide_features):
    return dst.data

def make_step(X, model, **kwargs):
    #     X = X.copy()

    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = kwargs.pop('lr', 5.0)
    max_jitter = kwargs.pop('max_jitter', 100)
    num_iterations = kwargs.pop('num_iterations', 100)
    show_every = kwargs.pop('show_every', 25)
    end_layer = kwargs.pop('end_layer', 3)
    object = kwargs.pop('objective', objective_L2)
    guide_features = kwargs.pop('guide_features', None)
    # print(end_layer)
    for t in range(num_iterations):
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        model.zero_grad()
        X_tensor = torch.Tensor(X)
        X_Variable = Variable(X_tensor.cuda(), requires_grad=True)

        act_value = model.forward(X_Variable, end_layer)
        diff_out = object(act_value, guide_features)
        act_value.backward(diff_out)

        learning_rate_ = learning_rate / np.abs(X_Variable.grad.data.cpu().numpy()).mean()

        X_Variable.data.add_(X_Variable.grad.data * learning_rate_)
        X = X_Variable.data.cpu().numpy()
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)
        X[0, :, :, :] = np.clip(X[0, :, :, :], -mean / std, (1. - mean) / std)

        if t == 0 or (t + 1) % show_every == 0:
            showtensor(X)

    return X


def dream(model, base_img, octave_n=4, octave_scale=1.4,
              end='', **step_params):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    octaves = [base_img]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail
        print(input_oct.shape)
        out = make_step(input_oct, model, **step_params)
        detail = out - octave_base