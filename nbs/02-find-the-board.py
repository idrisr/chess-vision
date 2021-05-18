from fastcore.transform import *
from fastai.data.all import *
from fastai.vision.all import *
import pandas as pd
from pathlib import Path
from nbdev import show_doc
from IPython.core.debugger import set_trace
from sklearn.metrics import jaccard_score
from PIL import Image, ImageDraw
import os

def get_y(o): 
    _,_,_,_,dx,dy=df.iloc[int(o.parts[-1].split('.')[0])]
    return [[dx, dy, 400+dx, 400+dy]]
def get_label(noop): return [None]

df = pd.read_csv('annotations.csv')
gen_url = Path(os.environ['HOME'] + "/.fastai/data/chess_screenshots2")

block = DataBlock(
    blocks=(ImageBlock, BBoxBlock, BBoxLblBlock), 
    get_items=get_image_files, 
    get_y=[get_y, get_label],
    n_inp=1,
    item_tfms=[Resize(224)])

block.c=1
dls=block.dataloaders(gen_url, batch_size=64)
dls.show_batch(max_n=4, figsize=(8, 8))
resnet34(pretrained=True)
