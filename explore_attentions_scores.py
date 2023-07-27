import torch
import sys
import time
import numpy as np
import fire
import json
import torchvision.datasets as dset
from captioning_model import CaptioningModel, setup_model_parallel
import os
from PIL import Image
import cv2

def get_attention_scores(model, test_dset, json_path):
    idx = test_dset.coco.getImgIds()[0]
    print('IMAGE IDX : {}'.format(idx))
    img = test_dset._load_image(idx)
    cap = test_dset.coco.imgToAnns[idx][0]['caption']
    # Get the logits
    #tokens = model.llama_tokenizer.encode(cap.lower(), bos=True, eos=True)

    cap_pred, tokens = model.generateCap(model.clip_preprocess(img).unsqueeze(0), 0, return_tokens=True)
    tokens = tokens[0]
    ca_scores = model.ca_scores.detach().cpu()
    print('Cross-attention shape : ', ca_scores.shape)
    #ca_scores = einops.reduce(ca_scores,'batch heads sequence img_features -> sequence img_features',

    attention_scores = model.att_scores.detach().cpu()
    print('Self-attention shape : ', attention_scores.shape)
    ca_map = {}
    att_map = {}
    for i in range(attention_scores.shape[1]):
        ca_map[i] = ca_scores[0, i, :, :].tolist()
        att_map[i] = attention_scores[0, i, :, :].tolist()
    print('Tokens : ', tokens)
    words = []
    for n in range(len(tokens)):
        if tokens[n]!=model.llama_tokenizer.eos_id:
            words.append(model.llama_tokenizer.decode(tokens[n]))
        else:
            break
    res = {'cross_attention':ca_map, 'self_attention':att_map, 'words':words, 'gt_cap':cap}
    print('Words : ', words)
    filename = '{}_{}.json'.format(json_path, idx)
    with open(filename, 'w') as f:
        json.dump(res, f)
    print('-- JSON file saved')



def main(model_path: str, nb_ca: int, p_test: float, temperature: float, json_path: str):
    """
    Main function of the inference script for the captioning model
    @param model_path: path to the weights of the fine-tuned captioning model
    @param nb_ca: number of added cross-attention layer in the fine-tuned captioning model
    @param p_test: proportion of the test dataset to use for inference
    @param temperature: temperature value to be used during inference of the captioning model
    @param json_path: path to the JSON file in which the generated captions will be stored
    @return:
    """
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
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len=512, max_batch_size=32, nb_ca=nb_ca
    )
    captioning_model.load_state_dict(torch.load(model_path))

    # Path to the coco caption dataset (val part)
    ROOT_val = "/users/rv2018/archive/data/img/train2014/val2014"
    FILE_val = "/users/rv2018/archive/data/annotations/annotations/captions_val2014.json"

    test_dset = dset.CocoCaptions(root=ROOT_val, annFile=FILE_val, transform=captioning_model.clip_preprocess)
    # Evaluation on test data
    print("INVESTIGATION ATTENTION SCORES")
    get_attention_scores(captioning_model, test_dset, json_path)
    # investigate_temperature(captioning_model, test_dset, 'eval_temp_effect')
    # get_attention_scores(captioning_model, test_dset, json_path)
    # img_path = './img_test/20230722_183856.jpg'
    # for t in [0.0, 0.1, 0.2, 0.4]:
    #     inference(img_path, captioning_model, t)
    #


if __name__ == "__main__":
    fire.Fire(main)
