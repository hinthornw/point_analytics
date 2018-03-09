'''Random utilities such as mapping to answers'''
import functools
from PIL import Image
from skimage import io


def bake_function(f, **kwargs):
    return functools.partial(f, **kwargs)


def blend_ims(im1, im2, mask):
    pass


def softmax(X, theta=1.0, axis=None):
    import numpy as np
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def map2each_answer(responses, f):
    for rsp in responses:
        output = rsp['output']
        for out in output:
            f(out)


def question2key(question):
    imid = list(question.keys())[0].strip()
    cls_inst_part = list(question[imid]['points']['data'].keys())[0]
    points = question[imid]['points']['data'][cls_inst_part]['points']
    x, y = str(points[0][0]), str(points[0][1])
    return imid + "_" + cls_inst_part + "_" + x + "_" + y


def separate_key(key):
    vals = key.rsplit("_", 5)
    return {
        'imid': vals[0],
        'obj_id': vals[1],
        'inst_id': vals[2],
        'part_id': vals[3],
        'xCoord': vals[4],
        'yCoord': vals[5]}


def get_im_from_s3(imid):
    if isinstance(imid, int):
        imid = str(imid)[:4] + "_" + str(imid)[4:]
    prefix = 'https://s3.amazonaws.com/visualaipascalparts/'
    suffix = '.jpg'
    uri = prefix + imid + suffix
    img = io.imread(uri)
    img = Image.fromarray(img)
    return img
