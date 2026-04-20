# NMN-BART 
NMN-BART is an architecture that integrates the compositional reasoning capabilities of Neural Module Networks (NMNs) with the text generation power of the pre-trained BART language model.

## Dataset
1. Downloaded VQA-X dataset:
```wget  http://images.cocodataset.org/zips/train2014.zip```

2. Download grounded image features:
```wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip```

3. Preprocess image features:
```python preprocess_features.py --input_tsv_folder /your/path/to/trainval_36/ --output_h5 /your/output/path/trainval_feature.h5```

## Installation
### Enviroment
1. Install torch
- ```pip install torch==1.10.1+cu113 torchvision -f https://download.pytorch.org/whl/cu113/torch_stable.html```
- ```pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-geometric==2.0.0 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html```
2. Install transformers
- ```pip install transformers==4.9.1 nltk spacy==2.1.6```
3. Evaluation
- ```pip install datasets```
- ```pip install evaluate```
- Build SPICE for evaluation following the [instruction](https://github.com/peteanderson80/SPICE).


## Run 
```python train_val.py```