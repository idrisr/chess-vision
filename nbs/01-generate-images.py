import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from random import randint

df = pd.read_csv("annotations.csv")
web_url = Path("/Users/id/.fastai/data/websites/")
board_url = Path("/Users/id/.fastai/data/kaggle-chess/")
gen_url = Path("/Users/id/.fastai/data/chess_screenshots2/")

def make_image(board, website, x, y, dx, dy):
    b = Image.open(board_url/board)
    s = Image.open(web_url/website)
    s.paste(b, (x, y))
    c = s.crop((x-dx, y-dy, x+400+dx, y+400+dy))
    return c

def save_image(img, i): img.save(gen_url / f'{i:06d}.png')
def PILrect(board, website, x, y, dx, dy): return ((dx, dy), (400+dx, 400+dy))

[save_image(make_image(*_[1]), _[0]) for _ in df.iterrows()]
