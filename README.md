# ProjectDeepLearning

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Shueyr/ProjectDeepLearning.git
cd ProjectDeepLearning
```

Use the provided conda environment to install all dependencies:

```bash
conda env create -f environment.yml
conda activate dl_project
```

### Dataset And Assets Setup

Use the notebook to download the dataset images and the FastText model:

- `setup_assets.ipynb`

The image download step can take a few tries depending on network stability and source availability, so rerun the relevant cells if needed.

## Usage

Train the model:

```bash
python ProjectDeepLearning/main.py --train
```

Evaluate the model:

```bash
python ProjectDeepLearning/main.py --evaluate
```
