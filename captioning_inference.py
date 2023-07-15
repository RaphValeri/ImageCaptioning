import torch.nn as nn
from pycocotools.coco import COCO
import torch
import sys
import time
import numpy as np
import fire
import random
from tempfile import TemporaryDirectory
import json

import torchvision.datasets as dset 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

from torch import autograd

from captioning_model import CaptioningModel, setup_model_parallel
import os
import cv2
from PIL import Image
import math


def predict_test_data(model, test_dset, filename="evaluation.json", temperature=0.6, p_test=1):
    data = []
    for i in range(int(p_test*len(test_dset.coco.getImgIds()))):
        if i in [int(0.1*j*p_test*len(test_dset.coco.getImgIds())) for j in range(1, 10)]:
            print("Inference {}/{} ...".format(i, int(p_test*len(test_dset.coco.getImgIds()))))
        idx = test_dset.coco.getImgIds()[i]
        img = test_dset._load_image(idx)
        pred = model.generateCap(model.clip_preprocess(img).unsqueeze(0), temperature)
        data_i = {"image_id":idx, "caption":pred[0]}
        data.append(data_i)
    with open(filename, 'w') as f:
        json.dump(data, f)
    print("End of prediction on test data")


def main(model_path : str, p_test : float, temperature : float, json_path : str):
    print("model path : ", model_path)
    print("temperature : ", temperature)
    print("json path : ", json_path)
    # Init setup
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # Load the model
    ckpt_dir = "download/7B"
    tokenizer_path = "download/tokenizer.model"

    captioning_model = CaptioningModel(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len=512, max_batch_size=32
    )
    captioning_model.load_state_dict(torch.load(model_path))

    #Path to the coco caption dataset (val part)
    ROOT_val = "/users/rv2018/files/MScProject/data/img/train2014/val2014"
    FILE_val = "/users/rv2018/files/MScProject/data/annotations/annotations/captions_val2014.json"

    
    test_dset = dset.CocoCaptions(root=ROOT_val, annFile=FILE_val, transform=captioning_model.clip_preprocess)
    # Evaluation on test data
    print("PREDICTION ON TEST DATA")
    predict_test_data(captioning_model, test_dset, json_path, temperature=temperature, p_test=p_test)






if __name__ == "__main__":
    fire.Fire(main)
