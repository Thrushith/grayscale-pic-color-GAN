import streamlit as st
import glob
import random
import os
import math
import itertools
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg19
from util import *
import torch
from col_generator import *

def upload_image():
    uploaded_file = st.file_uploader("Upload your Image here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        sample_image_path = './content/' + uploaded_file.name
        return sample_image_path

def load_generator_model1(epoch):
    generator = color_ecv()
    saved_model_path = "./generator.pth"
    state_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

def predict_outputs(model, dataset):
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device('cpu')
    model = model.to(device)
    Tensor = torch.Tensor
    outputs = {}
    for i, imgs in enumerate(dataloader):
        imgs_black = imgs["black"].type(Tensor)
        imgs_black_orig = imgs["orig"].type(Tensor)
        gen_ab = model(imgs_black)
        gen_ab = gen_ab.detach()
        gen_color = postprocess_tens_new(imgs_black_orig, gen_ab)[0].transpose(1, 2, 0)
        outputs[imgs["path"][0]] = gen_color
    return outputs

class TestDataset(Dataset):
    def __init__(self, root, single_image):
        if single_image:
            self.files = [root]
        else:
            self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        black_path = self.files[index % len(self.files)]
        img_black = np.asarray(Image.open(black_path))
        if(img_black.ndim==2):
            img_black = np.tile(img_black[:,:,None],3)
        (tens_l_orig, tens_l_rs) = preprocess_img(img_black, HW=(400, 400))
        return {"black": tens_l_rs.squeeze(0), 'orig': tens_l_orig.squeeze(0), 'path' : black_path}

    def __len__(self):
        return len(self.files)

def Dashboard():
    st.title("Upload Image here")
    uploaded_image = upload_image()
    sample_image_path = uploaded_image
    model = load_generator_model1(50)
    if uploaded_image is not None:
        single_image = True
        dataset = TestDataset(sample_image_path, single_image)
        outputs = predict_outputs(model, dataset)
        st.image(outputs[sample_image_path], caption="Colorized Image", use_column_width=True)