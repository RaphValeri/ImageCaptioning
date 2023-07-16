import os
import json
from pathlib import Path 
import cv2

from typing import Optional, Tuple 
import math 
import torch.nn.functional as F
import fire
import time 
import clip
import torch 
import torch.nn as nn

import torchvision.datasets as dset 
import torchvision.transforms as transforms
from PIL import Image

from llama import ModelArgs, Transformer, LLaMA, RMSNorm, apply_rotary_emb
from llama.tokenizer import Tokenizer
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


class CaptioningModel(nn.Module):
    def __init__(self, ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size):
        super().__init__()
        # Get the pretrained LLaMA and CLIP models
        generator, self.args = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
        self.llama_model = generator.model
        self.llama_tokenizer = generator.tokenizer
        self.freqs_cis = self.llama_model.freqs_cis
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_visual = clip_model.visual.to(self.device)
        # Freeze all their parameters
        for p in self.llama_model.parameters():
            p.requires_grad = False
        for p in self.clip_visual.parameters():
            p.requires_grad = False
        # Create a list with the custom layers of cross attention
        self.nb_ca = 1
        self.ca_layers = torch.nn.ModuleList()
        self.ca_norms = torch.nn.ModuleList()
        for i in range(self.nb_ca):
            self.ca_layers.append(CrossAttention(self.args))
            self.ca_norms.append(RMSNorm(self.args.dim, eps=self.args.norm_eps).to(device=self.device, dtype=torch.float))


    def forward(self, tokens, img, start_pos):
        with torch.autograd.enable_grad():
            # Embedding 
            verbose = False
            _bsz, seqlen = tokens.shape
            h = self.llama_model.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                #print("The mask is used !")
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            img = img.to(device=self.device, dtype=torch.half)
            img_features = self.clip_visual(img)
            # Forward pass to the Transformer blocks
            for i, layer in enumerate(self.llama_model.layers):
                #if i>=(len(self.llama_model.layers)-self.nb_ca):
                if i < self.nb_ca:
                    # Compute all the operation of a transformer block
                    if verbose:
                        print("--"*15)
                        print("iterqtion : ", i )
                        print("h dtype : ", h.dtype)
                    # Multihead self-attention
                    h = h + layer.attention.forward(layer.attention_norm(h), start_pos, freqs_cis, mask)
                    # Cross-Attention
                    #h = h + self.ca_layers[i - len(self.llama_model.layers) + self.nb_ca].forward(layer.ffn_norm(h), img_features, start_pos, freqs_cis, mask)
                    h = h + self.ca_layers[i].forward(layer.ffn_norm(h),img_features,start_pos, freqs_cis, mask)
                    h = h + layer.feed_forward.forward(self.ca_norms[i](h).to(dtype=torch.half))
                else:
                    h = layer(h, start_pos, freqs_cis, mask)
            h = self.llama_model.norm(h)
            output = self.llama_model.output(h[:, :, :])  # compute all logits
        return output.float()

    
    def generateCap(self, img, temperature=0, max_gen_len=30, top_p=0.95, verbose=False):
        """"
        Generation of a caption (inference of the model)
        """
        prompts = [""]
        bsz = len(prompts)
        prompt_tokens = [self.llama_tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.llama_tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.llama_tokenizer.pad_id
        start_pos = min_prompt_size
        for cur_pos in range(start_pos, total_len):
            logits = self.forward(tokens[:, 0:cur_pos], img, 0)
            logits = logits[:, -1, :]
            if verbose :
                print("Logit : ", logits)
                print("--"*15)
                print("Tokenizer world size : ", self.llama_tokenizer.n_words)
                print("--"*15)
            if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
            else :
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
        decoded = []
        if verbose :
            print("tokens : ", tokens.tolist())
        for i, t in enumerate(tokens.tolist()):
            if verbose:
                print("i : ", i)
                print("t : ", t)
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.llama_tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.llama_tokenizer.decode(t))
        if verbose:
            print("Prediction t={}: {}".format(temperature, decoded))
        return decoded




class CrossAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=self.device,
            dtype=torch.half,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=self.device,
            dtype=torch.half,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=self.device,
            dtype=torch.half,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            device=self.device,
            dtype=torch.half,
        )
        #self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

        #print("--"*15)
        #print("Dim : ", args.dim)
        #print("n_heads : ", args.n_heads)



    def forward(self, x: torch.Tensor, x_img: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        verbose = False

        bsz, seqlen, _ = x.shape
        if verbose :
            print("--"*15)
            print("Batch size : ", bsz)
            print("Sequence length : ", seqlen)
            print("x shape : ", x.shape)
            print("Initial x_img shape : ", x_img.shape)

        # Get the query from the language features and the keys and values from the image features
        x_img = x_img.repeat(1, seqlen, self.dim//x_img.shape[-1]) # Repeat the visual features to match the length of the token sequence
        if verbose:
            print("Reshape x_img : ", x_img.shape)
            print("--"*15)
            print("x type :", x.dtype)
            print("x_img type :", x_img.dtype)
            print("wq type :", self.wq.weight.dtype)
            print("--"*15)
        xq, xk, xv = self.wq(x), self.wk(x_img), self.wv(x_img)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)

        #scores = self.gate.tanh().half()*F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)


        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)




def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator, model_args

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    # Init setup
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    max_gen_len=30
    

    # Load dataset
    ROOT_train = "/users/rv2018/files/MScProject/data/img/train2014/train2014"
    FILE_train = "/users/rv2018/files/MScProject/data/annotations/annotations/captions_train2014.json"

    cap_train = dset.CocoCaptions(root=ROOT_train, annFile=FILE_train )

    print("--TRAINING DATA--")
    print("Number of samples : {}".format(len(cap_train)))
    img, target = cap_train[3]
    #img = Image.fromarray(cv2.imread("/users/rv2018/files/MScProject/llama/clip/COCO_train2014_000000000034.jpg"))
    print("IMAGE : ", img)
    verbose = False

    # Instantiate the captioning model
    print("--INSTANTIATION OF THE CAPTIONING MODEL--")
    captioning_model = CaptioningModel(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    print("--"*15)
    print("Trainable parameters : ", sum(p.numel() for p in captioning_model.parameters() if p.requires_grad))
    print("Total number of parameters : ", sum(p.numel() for p in captioning_model.parameters()))
    print("--"*15)
    print("--"*15)
    print("EOS Token : ", captioning_model.llama_tokenizer.eos_id)
    print("BOS Token : ", captioning_model.llama_tokenizer.bos_id)
    print("Pad Token : ", captioning_model.llama_tokenizer.pad_id)
    print(" 0 decoded : ", captioning_model.llama_tokenizer.decode([0]))
    print("--"*15)

    
    prompts = ["Two men are playing in the garden "]
    bsz = len(prompts)
    prompt_tokens = [captioning_model.llama_tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    min_prompt_size = min([len(t) for t in prompt_tokens])
    max_prompt_size = max([len(t) for t in prompt_tokens])

    total_len = min(captioning_model.args.max_seq_len, max_gen_len + max_prompt_size)

    tokens = torch.full((bsz, total_len), captioning_model.llama_tokenizer.pad_id).cuda().long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    input_text_mask = tokens != captioning_model.llama_tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    for cur_pos in range(start_pos, total_len):
        logits = captioning_model.forward(tokens[:, prev_pos:cur_pos], img, prev_pos)
        logits = logits[:, -1, :]
        if verbose :
            print("Logit : ", logits)
            print("--"*15)
            print("Tokenizer world size : ", captioning_model.llama_tokenizer.n_words)
            print("--"*15)
        if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
        else :
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
    decoded = []
    if verbose :
        print("tokens : ", tokens.tolist())
    for i, t in enumerate(tokens.tolist()):
        if verbose:
            print("i : ", i)
            print("t : ", t)
        # cut to max gen len
        t = t[: len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        try:
            t = t[: t.index(captioning_model.llama_tokenizer.eos_id)]
        except ValueError:
            pass
        decoded.append(captioning_model.llama_tokenizer.decode(t))
    print("Captioning model prediction : ", decoded)

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


if __name__ == "__main__":
    fire.Fire(main)