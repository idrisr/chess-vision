from fastcore.transform import *
from fastai.data.all import *
from fastai.vision.all import *

__all__ = ['NoLabelBBoxLabeler', 'BBoxTruth', 'iou']

class NoLabelBBoxLabeler(Transform):
    """ Bounding box labeler with no label """
    def setups(self, x): noop
    def decode (self, x, **kwargs):
        self.bbox,self.lbls = None,None
        return self._call('decodes', x, **kwargs)

    def decodes(self, x:TensorBBox):
        self.bbox = x
        return self.bbox if self.lbls is None else LabeledBBox(self.bbox, self.lbls)


class BBoxTruth:
    """ get bounding box location from DataFrame """
    def __init__(self, df): self.df=df
        
    def __call__(self, o):
        _,_,_,_,dx,dy=self.df.iloc[int(o.parts[-1].split('.')[0])]
        return [[dx, dy, 400+dx, 400+dy]]


def iou(pred, target):
    """ Vectorized Intersection Over Union calculation """
    target = Tensor.cpu(target).squeeze(1)
    pred = Tensor.cpu(pred)
    ab = np.stack([pred, target])
    intersect_area = np.maximum(ab[:, :, [2, 3]].min(axis=0) - ab[:, :, [0, 1]].max(axis=0), 0).prod(axis=1)
    union_area = ((ab[:, :, 2] - ab[:, :, 0]) * (ab[:, :, 3] - ab[:, :, 1])).sum(axis=0) - intersect_area
    return (intersect_area / union_area).mean()
