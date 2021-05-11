from random import randint
import random
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
from collections import namedtuple
from fastai.vision.utils import get_image_files
import matplotlib.pyplot as plt
import numpy as np

wbs =  get_image_files(Path("/Users/id/.fastai/data/websites/"))
bss = get_image_files(Path("/Users/id/.fastai/data/kaggle-chess/"))

def path(x): return '/'.join(x.parts[-2:])

df0 = pd.Series(bss, name="board")
df0 = df0.apply(path)

df1 = pd.Series(wbs, name="website")
df1 = df1.repeat(len(df0)//len(df1)+1)[:len(df0)]
df1 = df1.reset_index(drop=True)

np.random.seed(123)
df1=df1.sample(frac=1).reset_index(drop=True)
df1 = df1.apply(path)
df = pd.concat([df0, df1], axis=1, ignore_index=False)

def make_sample():
    w_width = 1024
    w_height = 768
    b_width=400
    b_height=400
    x = randint(50, w_width - b_width - 50)
    y = randint(50, w_height - b_height - 50)
    dx = randint(20, 100)
    dy = randint(20, 100)
    return (x, y, dx, dy)

random.seed(123)
df1 = pd.DataFrame([make_sample() for _ in range(len(df))], columns=['x', 'y', 'dx', 'dy'])
df = pd.concat([df, df1], axis=1)

df.to_csv('annotations.csv', index=False)
