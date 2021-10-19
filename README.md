# GALIROOT
DTU Special course on the GALIRUMI project.

## Setup

The dependecies of the repositories can be installed by the following:

`pip install -r requirements.txt`

The images are contained under *annotated_images.zip* and the transformed and split data can be found in the */pickle_jar*:

```
data
└──annotated_images.zip
    └──ann
    └──img
└──pickle_jar
    └──train_set.pickle
    └──valid_set.pickle
models
└──baseline_models.zip

```

## Usage

`train_pipeline.py` is running the training loop, based on the settings of `train_config.json` and `*_dataset.json`. [^1] It is using Tensorboard as a visualization tool. The batch size and epoch has to be manually changed as of now.

`inference_test.py` is a small script plotting individually 10 samples from the whole dataset and generating a 2-by-5 grid of it (saved to data/inference). The purpose is the visual analysis of the predictions.









[^1]: this part might be redundant, I will redo it later