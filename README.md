# DRO vs. Fairness
Project for the Machine Learning and Optimization course at RPI exploring the trade-offs between distributionally robust optimization (DRO) and fairness in machine learning.

Our presentation for this project can be found in `dro-vs-fairness-slides.pdf`.

Group Members: Roman Silen (silenr@rpi.edu), Jared Gridley (gridlj@rpi.edu), Dan Stevens (steved7@rpi.edu), Cole Mediratta (medirc@rpi.edu), Will Hawkins (hawkiw2@rpi.edu)


## Dataset
In this project, we use the CelebA Dataset (Liu et al., 2015), which can be downloaded [here](https://docs.google.com/u/0/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM).  In order to run this dataset with our code, download `img_align_celeba.zip` from the link and store it in the `/data/celeba/img_align_celeba/` directory.

To use a subset of the dataset, download the first 5000 images from `img_align_celeba.zip` into the `/data/celeba/img_align_celeba_subset/` directory.

## Code

### Prerequisites
The following python packages are used in the code: NumPy, Pandas, Matplotlib, PyTorch, Torchvision, Pillow (PIL)

### Using the Fairness GAN
See readme in FairnessModels folder.  
run `bash run_fairness.sh` in a bash shell

### Running the Experiments

The three parameters we are most concerned with are the optimization algorithm, dataset, and L2-penalty.  These can be changed using the `-t`, `--aug_data`, and `--l2_reg` flags, respectively.  Below are some examples of command line arguments to use.

- Train ResNet-18 using DRO with strong L2-penalty for 100 epochs
  - `python dro_fairness_exp.py -m resnet18 -t DRO --l2_reg 0.1 --model_save_file model_resnet18_DRO-1l2.pth --results_file results_resnet18_DRO-1l2.txt --accuracy_csv acc_resnet18_DRO-1l2.csv`

- Train ResNet-50 using ERM with standard regularization for 5 epochs on the augmented dataset
  - `python dro_fairness_exp.py -m resnet50 -e 5 --aug_data 1 --model_save_file model_resnet50_ERM-4l2.pth --results_file results_resnet50_ERM-4l2.txt --accuracy_csv acc_resnet50_ERM-4l2.csv`

For full list of command line arguments, see the `main()` function of `dro_fairness_exp.py` 

### Generating Plots

The data for accuracy during training is stored in a `.csv` file in the `/results/` directory.  To generate plots from this file, run the script `/helper_scripts/acc_plots.py` as follows:

- `python acc_plots.py -f [FILE]`

where `[FILE]` is just the file name alone (e.g. `acc_resnet50_ERM-4l2.csv`).

### Evaluating Model Performance

To obtain the performance metrics of a model saved in the `/models/` directory, use the script `/helper_scripts/performance.py`.  Specify the model file and model architecture as command line arguments, for example:
- `python performance.py -m model_resnet18_DRO-4l2.pth -a resnet18`
