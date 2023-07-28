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


def predict_test_data(model, test_dset, filename="evaluation.json", temperature=0.6, p_test=1.0):
    """
    Generation of the captions of images from a split of the MSCOCO captions dataset. Each caption with the ID of its
    corresponding image will be stored in a JSON file.
    @param model: the captioning model
    @param test_dset: split of the MSCOCO dataset
    @param filename: filename of the output JSON file containing all the generated captions
    @param temperature: value of the temperature hyperparameter to be used for inference of the captioning model
    @param p_test: proportion of the input test_dset dataset to use for inference
    @return:
    """
    data = []
    inf_time = []
    for i in range(int(p_test*len(test_dset.coco.getImgIds()))):
        if i in [int(0.1*j*p_test*len(test_dset.coco.getImgIds())) for j in range(1, 10)]:
            print("Inference {}/{} ...".format(i, int(p_test*len(test_dset.coco.getImgIds()))))
        # Prediction
        idx = test_dset.coco.getImgIds()[i]
        img = test_dset._load_image(idx)
        t0 = time.time()
        pred = model.generateCap(model.clip_preprocess(img).unsqueeze(0), temperature)
        # Store data
        tf = time.time() - t0
        inf_time.append(tf)
        data_i = {"image_id":idx, "caption":pred[0]}
        data.append(data_i)
    # Store the predicted captions in a json file
    with open(filename, 'w') as f:
        json.dump(data, f)
    # Store the inference times in a npy file
    time_path = filename.split('.json')[0] + 'inference_time.npy'
    with open(time_path, 'wb') as f:
        np.save(f, np.array(inf_time))
    print("End of prediction on test data")


def inference(img_path: str, model: CaptioningModel, temp: float, verbose : bool = True) -> str:
    """
    Single inference of the captioning model for an image
    @param img_path: path of the image
    @param model: captioning model
    @param temp: temperature value to be used during inference
    @param verbose: boolean to print or not the generated caption
    @return: Generated caption
    """
    img = Image.fromarray(cv2.imread(img_path))
    cap = model.generateCap(model.clip_preprocess(img).unsqueeze(0), temp)
    if verbose:
        print('- Inference for image {} -'.format(img_path))
        print('t={} | {}'.format(temp, cap))
    return cap


def investigate_temperature(model, test_dset, json_path, temps=[0.1, 0.3, 0.5, 0.7, 0.9], n_best=5):
    idx = test_dset.coco.getImgIds()[0]
    print('IMAGE IDX : {}'.format(idx))
    img = test_dset._load_image(idx)
    cap = test_dset.coco.imgToAnns[idx][0]['caption']
    # Get the logits
    tokens = model.llama_tokenizer.encode(cap.lower(), bos=True, eos=True)
    logits = model(torch.tensor(tokens).cuda().long().view(1, -1), model.clip_preprocess(img).unsqueeze(0), 0)

    res = {}
    for n in range(logits.shape[1]):

        string_input = model.llama_tokenizer.decode(tokens[n])
        string_eval = {}
        l_n = logits[:, n, :]
        for t in temps:
            probs_t = {}
            # Investigate the use of temperature
            if t>0:
                probs = torch.softmax(l_n / t, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

                best_probs = probs_sort[0, :n_best].tolist()
                best_idx = probs_idx[0, :n_best].tolist()

                for i in range(n_best):
                    probs_t[model.llama_tokenizer.decode(best_idx[i])] = best_probs[i]
                if string_input != '':
                    string_eval[t] = probs_t
        if string_input != '':
            res[string_input] = string_eval
    print(res)
    filename = '{}_{}.json'.format(json_path, idx)
    with open(filename, 'w') as f:
        json.dump(res, f)


def get_attention_scores(model, test_dset, json_path):
    idx = test_dset.coco.getImgIds()[0]
    print('IMAGE IDX : {}'.format(idx))
    img = test_dset._load_image(idx)
    cap = test_dset.coco.imgToAnns[idx][0]['caption']
    # Get the logits
    tokens = model.llama_tokenizer.encode(cap.lower(), bos=True, eos=True)

    cap_pred = model.generateCap(model.clip_preprocess(img).unsqueeze(0), 0)
    ca_scores = model.ca_scores.detach().cpu()
    #ca_scores = einops.reduce(ca_scores,'batch heads sequence img_features -> sequence img_features',

    attention_scores = model.att_scores.detach().cpu()

    ca_map = {}
    att_map = {}
    for i in range(attention_scores.shape[1]):
        ca_map[i] = ca_scores[0, i, :, :].tolist()
        att_map[i] = attention_scores[0, i, :, :].tolist()

    words = []
    for n in range(len(tokens)):
        if tokens[n]!=model.llama_tokenizer.eos:
            words.append(model.llama_tokenizer.decode(tokens[n]))
        else:
            break
    res = {'cross_attention':ca_map, 'self_attention':att_map, 'words':words, 'gt_cap':cap}

    filename = '{}_{}.json'.format(json_path, idx)
    with open(filename, 'w') as f:
        json.dump(res, f)



def main(model_path : str, nb_ca : int, p_test : float, temperature : float, json_path : str):
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

    #Path to the coco caption dataset (val part)
    ROOT_val = "/users/rv2018/archive/data/img/train2014/val2014"
    FILE_val = "/users/rv2018/archive/data/annotations/annotations/captions_val2014.json"

    
    test_dset = dset.CocoCaptions(root=ROOT_val, annFile=FILE_val, transform=captioning_model.clip_preprocess)
    # Evaluation on test data
    print("PREDICTION ON TEST DATA")
    predict_test_data(captioning_model, test_dset, json_path, temperature=temperature, p_test=p_test)
    print('PREDICTIONS ON EVERYDAY LIFE PICTURES')
    img_tests = os.listdir('img')
    for img_path in img_tests:
        if len(img_path.split('2023')) != 1:
            inference(os.path.join('img', img_path), captioning_model, 0.0)
            inference(os.path.join('img', img_path), captioning_model, 0.1)
            inference(os.path.join('img', img_path), captioning_model, 0.2)
            inference(os.path.join('img', img_path), captioning_model, 0.3)
    #investigate_temperature(captioning_model, test_dset, 'eval_temp_effect')
    #get_attention_scores(captioning_model, test_dset, json_path)
    # img_path = './img_test/20230722_183856.jpg'
    # for t in [0.0, 0.1, 0.2, 0.4]:
    #     inference(img_path, captioning_model, t)
    #




if __name__ == "__main__":
    fire.Fire(main)
