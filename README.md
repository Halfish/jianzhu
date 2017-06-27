image-text matching

refer to [Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/abs/1411.2539)

## build up envirnment
```python
# install gensim
pip install gensim

# pip install hyperboard
# refer to https://github.com/WarBean/hyperboard

# install pytorch
# refer to https://github.com/pytorch/pytorch

pip install torchvision

```

## how to train an image-text model?

### Step 1: Prepare dataset
wrap images and sentences in `datautil.py`

### Step 2: Train the model
The model was specified in `model.py`, run 
```shell
CUDA_VISIBLE_DEVICES=2 python main.py
```
to train an encoder-decoder model, we will get `encoder.pt`, then move it to models directory:
```shell
>>> mv encoder.pt static/models/
```

### Step 3: dump static images and sentences
dump static images and sentences for web server retrieval, run
```shell
CUDA_VISIBLE_DEVICES=2 python dump_static_data.py
```
you will get `image_static.npy` and `sentence_static.npy`, then move them to models directory,
```shell
>>> mv *_static.npy static/models/
```

### Step 4: startup tornado web server
```shell
python server.py
```
