# Image Captioning 

This repository is an implementation of a captioning model using two pre-trained models the CLIP visual encoder and the
LLaMA language model with 7B parameters.
In order to download the checkpoints and tokenizer of the LLaMA model, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5).

## Setup

In a conda env with pytorch / cuda available, run:
``` bash
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
$ pip install -r requirements.txt
```
Then in this repository:
``` bash
$ pip install -e .
```

## Download of LLaMA checkpoints and tokenizer

Follow the instructions from [LLaMA official repository](https://github.com/facebookresearch/llama/tree/llama_v1) to download 
the checkpoints and tokenizer after being approved.
Please save these files in a '/download' folder.

## Download of the MSCOCO captions dataset
To train the captioning model, download the MSCOCO captions dataset, for example for the 2014 train split you can run
the following instructions.

In the folder you want to store the images of the dataset, run:
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip
```

In the folder you want to store the annotations of the dataset, run:
```  bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip
```

You can do the same with the 'val2014' split for inference purpose (the 'test2014' split doesn't have the annotations).

Finally, edit the path toward your images and annotations in `captioning_training.py` for the training dataset and in 
`captioning_inference.py` for the testing set.
##  Training

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |



## Reference

This fine-tuned captioning model uses two pre-trained models: the LLaMA language model and the CLIP visual encoder.



LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```


## License
License of the LLaMA model: see the [LICENSE](LICENSE) file.
