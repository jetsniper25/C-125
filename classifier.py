import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
import PIL.ImageOps
import os,ssl,time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y=fetch_openml("mnist_784", version=1, return_X_y=True)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xtrainscale=xtrain/255.0
xtestscale=xtest/255.0

clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscale,ytrain)

def get_predict(img):
        im_pil=Image.open(img)
        Imagebw=im_pil.convert("L")
        Imagebw_resize=Imagebw.resize((28,28),Image.ANTIALIAS)
        pixelfilter=20
        minpixel=np.percentile(Imagebw_resize,pixelfilter)
        Imagebw_resize_scaled=np.clip(Imagebw_resize-minpixel,0,255)
        maxpixel=np.max(Imagebw_resize) 
        Imagebw_resize_scaled=np.asarray(Imagebw_resize_scaled)/maxpixel
        testsample=np.array(Imagebw_resize_scaled).reshape(1,784)
        testpred=clf.predict(testsample)
        