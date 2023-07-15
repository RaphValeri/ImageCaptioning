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

def investigate_temperature(model, test_dset, json_path, temps=[0.1, 0.3, 0.5, 0.7, 0.9], n_best=5):
    idx = test_dset.coco.getImgIds()[0]
    print('IMAGE IDX : {}'.format(idx))
    img = test_dset._load_image(idx)
    cap = test_dset.coco.imgToAnns[idx][0]['caption']
    # Get the logits
    tokens = model.llama_tokenizer.encode(cap.lower(), bos=True, eos=True)
    logits = model(torch.tensor(tokens).cuda().long().view(1, -1), model.clip_preprocess(img).unsqueeze(0), 0)
    #logits shape : (bz, nb_tokens, n_words)
    #print('Logits shape : ', logits.shape)
    #print('Nb tokens : ', logits.shape[1])
    res = {}
    for n in range(logits.shape[1]):
        #print('Token input : ', tokens[n])
        #print('Input : ', model.llama_tokenizer.decode(tokens[n]))
        string_input = model.llama_tokenizer.decode(tokens[n])
        string_eval = {}
        probs_t = {}
        l_n = logits[:, n, :]
        for t in temps:
            # Investigate the use of temperature
            if t>0:
                probs = torch.softmax(l_n / t, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                #print('Probs sorted shape : ', probs_sort.shape)
                #print('Probs idx sorted shape : ', probs_idx.shape)
                best_probs = probs_sort[0, :n_best].tolist()
                best_idx = probs_idx[0, :n_best].tolist()
                #print('Best probs : ', best_probs)
                #print('Best idx : ', best_idx)
                for i in range(n_best):
                    probs_t[model.llama_tokenizer.decode(best_idx[i])] = best_probs[i]
                if string_input !='':
                    string_eval[t] = probs_t
            #print('     t={} : {}'.format(t, probs_t))
        if string_input !='':
            res[string_input] = string_eval
    print(res)
    with open(json_path, 'w') as f:
        json.dump(res, f)


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
    #print("PREDICTION ON TEST DATA")
    #predict_test_data(captioning_model, test_dset, json_path, temperature=temperature, p_test=p_test)
    investigate_temperature(captioning_model, test_dset, 'eval_temp_effect.json')





if __name__ == "__main__":
    fire.Fire(main)
