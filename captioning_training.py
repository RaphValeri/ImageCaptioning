import torch.nn as nn
import torch
import sys
import time
import numpy as np
import random
from tempfile import TemporaryDirectory
import torchvision.datasets as dset 
from torch.utils.data import DataLoader
import fire
from captioning_model import CaptioningModel, setup_model_parallel
import os
import math


def custom_collate(batch):
    """
    Custom implementation of the collate function to tokenize the captions

    """
    img = torch.stack([item[0] for item in batch])
    # Randomly select one caption from the ground truth captions
    target = [item[1][random.randint(0, len(item[1])-1)] for item in batch]  
    tokens, labels = preprocess_captions(target)
    return img, tokens[:, :-1], labels[:, 1:]


def preprocess_captions(target_list):
    verbose=False
    # Clen the captions : remove dots and upper letters
    for i in range(len(target_list)):
        if target_list[i][-1]==".":
            cap = target_list[i][:-1]
            target_list[i] = cap.lower()
        else:
            cap = target_list[i][:]
            target_list[i] = cap.lower()
    # Tokenize the captions and pad them
    bsz = len(target_list)

    prompt_tokens = [captioning_model.llama_tokenizer.encode(x, bos=True, eos=True) for x in target_list]
    max_prompt_size = max([len(t) for t in prompt_tokens])
    #total_len = min(captioning_model.args.max_seq_len, max_gen_len + max_prompt_size)
    tokens = torch.full((bsz, max_prompt_size), 0).cuda().long()
    labels = torch.full((bsz, max_prompt_size), -100).cuda().long()
    if verbose:
        print("Target list : ", target_list)
        print("Prompt tokens : ", prompt_tokens)
        print("Max prompt size : ", max_prompt_size)
        #print("Total len : ", total_len)
    for k, t in enumerate(prompt_tokens):
        if verbose:
            print("k : ", k)
            print("t : ", t)
        tokens[k, : len(t)] = torch.tensor(t).long()
        labels[k, : len(t)] = torch.tensor(t).long()

    return tokens, labels



def train(model: nn.Module, optimizer, train_dataloader, loss_fct, lr, epoch, scheduler=None, verbose=False):
    # Turn the model in training mode
    model.train()  
    log_interval = 400
    total_loss = 0.0
    start_time = time.time()
    loss_train = []

    num_batches = len(train_dataloader)
    
    
    #with torch.autograd.set_detect_anomaly(True):
    for idx, batch in enumerate(train_dataloader):

        # Get the images and tokenized captions for the batch
        img_batch, tokens, labels = batch[0], batch[1], batch[2]

        prev_pos = 0
        # Compute the prediciton of the model
        output = model(tokens, img_batch, prev_pos)
        if idx==0 and epoch==1:
            print("Output size : ", output.permute(0,2,1).size())
            print("Labels size : ",  labels.size())
        # Compute the loss
        loss = loss_fct(output.permute(0,2,1), labels)

        # Clear the values of the gradients
        optimizer.zero_grad()

        # Compute the gradients
        loss.backward()

        # Clip the gradient
        #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)

        # Update the weights
        optimizer.step()


        # Update the value of the total loss
        total_loss += loss.item()
        if idx % log_interval == 0 and idx>0:
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {idx:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            loss_train.append(cur_loss)
            total_loss = 0
            start_time = time.time()	
        
        # print('epoch end')

    return sum(loss_train)/len(loss_train)
        




def evaluate_single(model: nn.Module, img_ref, temperature=0):
    model.eval()  # turn on evaluation mode
    with torch.no_grad():
        model.generateCap(img_ref.unsqueeze(0), temperature=temperature)
    model.train()


def evaluate(model: nn.Module, val_dataloader, loss_fct) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0
    with torch.no_grad():
       for idx, batch_val in enumerate(val_dataloader):
            img_batch, tokens, labels = batch_val[0], batch_val[1], batch_val[2]
            prev_pos = 0
            # Compute the prediciton of the model	 
            output = model(tokens, img_batch, prev_pos)
            # Compute the loss and update the gradient
            loss = loss_fct(output, labels)
            total_loss += loss.item()
    return total_loss/len(val_dataloader)




def main(epochs : int, loss_save_path : str, model_path : str):
    print("epochs : ", epochs)
    print("loss path : ", loss_save_path)
    print("model path : ", model_path)

    #Path to the coco caption dataset (train part)
    ROOT_train = "/users/rv2018/archive/data/img/train2014/train2014"
    FILE_train = "/users/rv2018/archive/data/annotations/annotations/captions_train2014.json"


    # Create dataset and dataloader
    bz = 16			# batch size
    p_train = 0.8	# proportion of data used for training
    p_val = 1 - p_train	 # proportion of data used for validation
    #p_test = 0.001	 # proportion of data used for test

    #coco = COCO(FILE_train)
    cap_dset = dset.CocoCaptions(root=ROOT_train, annFile=FILE_train, transform=captioning_model.clip_preprocess)
    generator = torch.Generator().manual_seed(42)
    train_dset, val_dset = torch.utils.data.random_split(cap_dset, [p_train, p_val], generator)
    # train_dset = torch.utils.data.Subset(train_dset, [i for i in range(bz)])
    # val_dset = torch.utils.data.Subset(val_dset, [i for i in range(bz)])

    train_dataloader = DataLoader(train_dset, batch_size=bz, shuffle=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dset, batch_size=bz, shuffle=True, collate_fn=custom_collate)


    # Display image and label.
    """
    train_features, train_tokens, train_labels = next(iter(train_dataloader))
    print(f"Feature batch size: {len(train_features)}")
    print(f"Labels batch size: {len(train_tokens)}")
    print("--"*15)
    img = train_features[0]
    tokens = train_tokens
    label = train_labels
    print("		Example of a training sample ")
    print("Batch of images shape : ", train_features.shape)
    print("Image : ", img)
    print("Tokens : ", tokens)
    print("\n")
    print("Labels : ", label)
    """
    print("--"*15)
    print("Trainable parameters : ", sum(p.numel() for p in captioning_model.parameters() if p.requires_grad))
    print("Total number of parameters : ", sum(p.numel() for p in captioning_model.parameters()))
    print("--"*15)

    best_val_loss = float('inf')
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, captioning_model.parameters()), lr=lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-04)
    print('start')
    loss_train = []
    loss_val = []

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            # print('start')
            epoch_start_time = time.time()
            loss_e = train(captioning_model, optimizer, train_dataloader, criterion, lr, epoch)
            val_loss = evaluate(captioning_model, val_dataloader, criterion)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            loss_train.append(loss_e)
            loss_val.append(val_loss)
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}'
                )
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(captioning_model.state_dict(), best_model_params_path)

           #scheduler.step()
        captioning_model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    loss_train = np.array(loss_train) 
    loss_val = np.array(loss_val)
    loss = np.array([loss_train, loss_val])
    with open(loss_save_path, 'wb') as f:
        np.save(f, loss)
    torch.save(captioning_model.state_dict(), model_path)
    print("- Captioning model saved as {}".format(model_path))
    print("- Loss evolution saved as {}".format(loss_save_path))
    


if __name__=="__main__":
    # Global variables
    max_gen_len = 30


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


    # Run the main function
    fire.Fire(main)

