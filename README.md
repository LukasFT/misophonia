# Misophonia ANC dataset and model

TODO: Add overall project description here.

Authors: Lukas Frimer Thorlander, Tonio Ermakoff.

## Getting Started

To get started easily, we recommend using the provided DevContainer configuration. This allows you to set up a consistent development environment with all necessary dependencies. To do this, ensure you have the following installed:

- Docker
- VS Code
- DevContainer Extension

## Misophonia Dataset

The first large-scale, open-access binaural dataset for misphonia trigger selective Active Noise Cancellation. This repo contains the pipeline used to generate on-the-fly, binaural misophonia mixtures, as well as the ```canonical v-1 dataset``` of 30k samples. Reproduction of our work, or use of the pipeline for original endeavors is described below. The pipeline includes source data from FSD50K, ESC-50, and FOAMS datasets. Additional source data can be introduced by creating a class that abides by the `SourceData` interface in `misophonia_dataset/interface.py`.

A detailed report of this project is included in `Misophonia_Dataset_Report.pdf`.


### Using the dataset in Python

To use the dataset in Python, you have two options. You can use the canonical version with the `PremadeMisophoniaDataset` class. Or you can generate your own on the fly using the `MisophoniaDatasetGenerator` class.


#### Initializing the canonical dataset

Currently, we have not set up a way to distribute the dataset efficiently (TODO). Please contact misophonia.dataset@lftm.org to get access to the data files.

```python
from misophonia_dataset.misophonia_dataset import PremadeMisophoniaDataset

dataset_name = "canonical-v1"  # Or use "demo-v1" for the small sample dataset distributed with this repo
dataset = PremadeMisophoniaDataset(dataset_name, base_save_dir="path/to/data/dir")
split = dataset.get_split("train")
```


#### Initializing a dataset generated on-the-fly

```python
from misophonia_dataset.source_data.esc50 import Esc50Dataset
from misophonia_dataset.source_data.foams import FoamsDataset
from misophonia_dataset.source_data.fsd50k import Fsd50kDataset
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset

data_dir = "path/to/data/dir"
source_data = (
    Esc50Dataset(save_dir=data_dir),
    FoamsDataset(save_dir=data_dir),
    Fsd50kDataset(save_dir=data_dir),
)

for source in source_data:
    # Make sure the data is in the data_dir
    # Can also be done using the CLI (see below)
    source.download_data()

dataset = GeneratedMisophoniaDataset(source_data=source_data)
split = dataset.get_split("train", num_samples=10)  # See doctring for more details on options
```

Note: You can also use the CLI or the `PremadeMisophoniaDataset.save_split` method to generate and save a custom dataset to disk for later use.


#### Iterating over dataset items

Using either of the two options above, you can iterate over the dataset as follows:

```python
for item in split:
    print(item)  # See details about this in misophonia_dataset.interface.MisophoniaItem

    is_trigger = item.is_trigger  # True if the mixture contains misophonia trigger sounds
    mix = item.get_mix_audio()  # A numpy array with binaural audio
    ground_truth = item.get_ground_truth_audio()  # A numpy array of the same dimentionality with only the binaural trigger sounds (if any)

    # Your own logic to train a model to predict the isolated triggers (ground_truth) from the entire mix (mix)
```

### Using the dataset CLI

You can use the CLI to run scripts. See more details by running:

```bash
python -m misophonia_dataset.main --help
```

#### Downloading Source Data
To download all the source data used to generate the dataset, run:

```bash
python -m misophonia_dataset.main download
```

This may take a while.

#### Reproducing the Canonical Dataset

To reproduce the canonical dataset splits, first download the source data as descibed above, then run the following commands:

```bash
python -m misophonia_dataset.main generate canonical-v1-reproduced test -n 3000 --seed 42 --add-experimental-pairs
python -m misophonia_dataset.main generate canonical-v1-reproduced val -n 7000 --seed 42
python -m misophonia_dataset.main generate canonical-v1-reproduced train -n 20000 --seed 42
```



## Misophonia ANC model
Training and evaluation of the selective ANC model are handled through the model CLI.

For detailed options and arguments, run:

```bash
python -m misophonia_anc.main preprocess --help
python -m misophonia_anc.main train --help
python -m misophonia_anc.main evaluate --help
python -m misophonia_anc.main cp_best_epoch --help
python -m misophonia_anc.main visualize_data --help
```

### Preprocessing

To preprocess the dataset into `data/model-v1` for training, run the following command from the project root:

```bash
python -m misophonia_anc.preprocess model-v1 --split train --split val --split test
```

A `config.yaml` file is required in `data/model-v1/`. This configuration file defines:

- Dataset generation parameters:
  - Number of binaural mixes per split
  - Trigger-to-control ratio
  - Desired SNR range
  - Number of background sounds per mix
  - Trigger subtraction methods for evaluation
- Training settings
  - Number of epochs
  - Batch size
  - Loss function
  - Trigger subtraction methods
- Model parameters
- Model hyperparameters

See `data/model-v1/config.yaml` for an example configuration.

Generated datasets for each split are compressed and saved under:

```text
data/model-v1/webdataset/<split-name>
```

### Training

Once the dataset has been generated and with a ready `config.yaml`, training can be started with:

```bash
python -m misophonia_anc.train model-v1
```

All major training settings are controlled through `config.yaml`.

For implementation details, see:

- `train_model` in `misophonia_anc/train.py`
- `MisophoniaANCNet` in `misophonia_anc/model.py`

At the end of each epoch, a checkpoint containing model weights and metadata is saved to:

```text
data/model-v1/checkpoints
```

If training is interrupted (for example during epoch 6), training can be resumed from epoch 5 with:

```bash
python -m misophonia_anc.train model-v1 --checkpoint weights_epoch_5.pt
```

Additional options:

- `--resume-mlflow` — Resume logging to an existing MLflow experiment
- `--reset-epoch` — Reset the epoch counter to `0`
- `--skip-subtraction` — Skip evaluation metrics based on subtraction methods

Subtraction methods are configured in `config.yaml` and implemented in:

```text
misophonia_anc/subtraction_methods.py
```

Using `--skip-subtraction` during training is generally not recommended.

### Evaluation

TODO