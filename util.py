import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import numpy as np
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def showtensor(a):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std.reshape([1, 1, 3]) * inp + mean.reshape([1, 1, 3])
    inp *= 255
    showarray(inp)
    clear_output(wait=True)