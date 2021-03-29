# Efficient detection of longitudinal bacteria fission using transfer learning in Deep Neural Networks

Deep learning based approach to predict longitudinal bacteria fission.<br>
The project train a binary classifier to predict the type of division on bacteria.

### Prerequisites
* git (optional, https://git-scm.com/)
* Nvidia driver (https://www.nvidia.com)
* cuda version 10 or higher (https://developer.nvidia.com/cuda-10.0-download-archive)
* Docker (optional, https://docs.docker.com/get-docker/)
* GPU

## Dataset Description.
The dataset is a collection of phase contrast microscopic images showing bacteria with a variety of shapes. The microscopic images are from the Thiosymbion species. More details about the images can be found in [Weber et al. 2019](https://doi.org/10.1016/j.cub.2019.07.064), [Leisch et al. 2016](https://doi.org/10.1038/nmicrobiol.2016.182) and [Pende et al. 2014](https://doi.org/10.1038/ncomms5803).<br>
**NOTE:** Instructions to create a dataset from microscopic images can be found [here](create_training_dataset.md)

## Ready to use (Docker Optional)
These instructions will get you running a traning, testing and class prediction of the current project.<br>
The easiest way to run the project is trough a Docker image. Follow the instructions [here](docker.md) to install Docker in your local machine.<br>

* Download one time the image from Docker Hub. In a terminal copy and paste the following line:<br>
`docker pull charlos1204/ldbc:gpu`

* Run a training: The downloaded image contains the dataset to train and test a new model.<br>
The options for a training are: train a model with internal dataset or train with your data.<br>
**The new model will be safe as "trained_net_sgd_pre-25.pth" in "current_dir" which is where you are running the docker image.**<br>
**$PWD** is the current directory (linux/macOS). For Windows replace with (include quotes) "C:\path_to_the_project_folder\"<br>
Copy and paste.<br>
   1. Train a new model with the internal dataset:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python train_model.py internal`
   2. Train a new model with external dataset ([follow this instructions to create a new dataset folder](create_training_dataset.md)):
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python train_model.py external /current_dir/your_dataset_folder`

* Run a test on the trained model: test the model on unseen data.<br>
  Copy and paste.<br>
   1. Test the internal model with the internal dataset:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python test_model.py internal_model`
   2. Test the internal model with a external dataset:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python test_model.py internal_model /current_dir/your_dataset_folder`
   3. Test the new model with the internal dataset:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python test_model.py external_model /current_dir/trained_net_sgd_pre-25.pth`
   4. Test the new model with a external dataset:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python test_model.py external_model /current_dir/trained_net_sgd_pre-25.pth /current_dir/your_dataset_folder`

* Predict a class of bacteria division:<br>
   Copy and paste.<br>
   1. Predict the class with the internal model on internal dataset of images:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python predict_class.py internal_model`
   2. Predict the class with the internal model on external dataset of images: create a new folder and add some bacteria images. Or use the samples folder from the current project by dowloding into the folder were you are running the docker image.
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python predict_class.py internal_model /current_dir/samples`
   3. Predict the class with the new model on internal dataset of images:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python predict_class.py external_model /current_dir/trained_net_sgd_pre-25.pth`
   4. Predict the class with the new model on external dataset of images:<br>
`docker run -ti -v $PWD:/current_dir/ charlos1204/ldbc:gpu python predict_class.py external_model /current_dir/trained_net_sgd_pre-25.pth /current_dir/samples`

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for testing purposes.

The following libraries can be install with miniconda or pip.
* miniconda (optional, https://docs.conda.io/en/latest/miniconda.html)
* python 3.6 or higher (https://www.python.org/)
* cython (https://cython.org/)
* opencv (https://docs.opencv.org/master/index.html)
* tqdm (https://github.com/tqdm/tqdm)
* torchsummary (https://github.com/sksq96/pytorch-summary)
* matplotlib (https://matplotlib.org/stable/index.html#)
* scipy (https://www.scipy.org/index.html)
* pytorch (https://pytorch.org/)
* JupyterLab (https://jupyter.org/)
* torch, torchvision (https://pytorch.org/get-started/locally/)
* sklearn (https://scikit-learn.org/stable/index.html)
* numpy (https://numpy.org/)
* pandas (https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html)

#### 1. Git for cloning the project (optional)
* To install git in Debian based distribution (Ubuntu):<br>
`sudo apt install git-all` 
* To install git in Windows download the binary from [here](https://git-scm.com/download/win)
* To install git in macOS download the binary from [here](https://git-scm.com/download/mac)

#### 2. Nvidia driver and cuda for the GPU card
* To install Nvidia driver for Linux, macOS, or Widnows will depend on the card you have, you can check the steps [here](https://www.nvidia.com/Download/index.aspx?lang=en-us), and for the cuda library [here](https://developer.nvidia.com/cuda-10.0-download-archive), we recommend asking your IT administrator for help.

## Deep learning libraries
The easiest way to install python and the libraries to run the code is through miniconda. Nevertheles, linux distributions already have python and pip installed. Pip is the package installer for python. If you prefer try with miniconda, follow the step 4. Otherwise skip to step 5 to use pip.

#### 3. Install libraries with miniconda
miniconda is a minimal installer for conda. conda includes the most recent python version and provides a way to install packages, such as the packages needed to do machine learning or deep learning.
* First download the miniconda installer from [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers) depending on your operating system.
* For linux run the following command:<br>
`bash Miniconda3-latest-Linux-x86_64.sh`
* For Windows 10: double-click the .exe file and follow the instructions on the screen.
* For macOS run the following command:<br>
`bash Miniconda3-latest-MacOSX-x86_64.sh`

Install libraries to run the code. In a terminal run the following commands:<br>
`conda install -c anaconda cython`<br>
`conda install -c conda-forge opencv`<br>
`conda install -c conda-forge tqdm`<br>
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`<br>
`conda install -c conda-forge pytorch-model-summary`<br>
`conda install -c conda-forge matplotlib`<br>
`conda install -c anaconda scipy`<br>
`conda install -c anaconda pandas`<br>
`conda install -c anaconda joblib`<br>
`conda install -c anaconda scikit-learn`<br>
`conda install -c conda-forge jupyterlab`<br>

#### 4. Install libraries with pip
* Linux (Ubuntu): In a terminal type the following command to check the python version installed in your operating system (most linux flavors will run the command):<br>
`python --version` or `$ python3 --version`
If the out put is: `Python 3.6.x` or higher you have the right version to run the code, follow the next step. Otherwise, install at least `Python 3.6`. For instructions to install `Python 3.6` click [here](install_python_linux.md).
* To install `Python 3.6` in Windows 10 download the binary from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe). Double-click the .exe file and follow the instructions on the screen.
* To install `Python 3.6` in macOS download the binary from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg), follow the instructions on the screen.

Install libraries to run the code. In a terminal run the following commands:<br>
`pip install cython`<br>
`pip install opencv-python`<br>
`pip install tqdm`<br>
`pip install torchsummary`<br>
`pip install matplotlib`<br>
`pip install pandas`<br>
`pip install scipy`<br>
`pip install joblib`<br>
`pip install scikit-learn`<br>
`pip install jupyterlab`<br>
`pip install torch torchvision torchaudio`


## Installation
1. Clone the project by running the following command in a terminal:<br>
`git clone https://github.com/charlos1204/longitudinal_division_classification.git`<br>
or download the code as a zip with the green button in the right upper corner.
2. Change to folder longitudinal_division_classification:<br>
`cd longitudinal_division_classification` or `cd longitudinal_division_classification-main`<br>
Content folder list:
    * model: folder to save the trained model.
    * data.zip: zip file for the training, testing and validation images.
    * Longitudinl_classification.ipynb: jupyter notebook to run the training code.
    * train_functions_sgd.py: python code with additional functions to run the training code.
    * README.md: current redme file
    * install_python_linux.md: python 3.6 installation instructions.
    * samples: sample images that the model has not seen during training. This images are for predicting the class.
    * fake_samples: fake samples images of bacteria to predict the class. This images are to see how good the model has generalized.
3. Unzip the data.zip file.
data folder list:
    * train: images of the two clases to train del model (longitudinal division and other division).
    * val: images of the two clases to validate the training (longitudinal division and other division).
    * test: testing images after the training the model (longitudinal division and other division).
Each folder contain two subfolders:
    * longitudinal_division: images of vertical spliting bacteria.
    * other_division: all other images of bacteria.

## Run a training for longitudinal division classification
Run the jupyter notebook in a terminal:<br>
`jupyter notebook Longitudinl_classification.ipynb`

**Notes: train_functions_sgd.py code contains all functions that are called in the main function. Do not remove or delete this file.**

## Contributing
We thank Philipp M. Weber, from the University of Vienna for providing the microscopic images, and Gabriela F. Paredes for her comments and insights on the review of the manuscript.

## Versioning
0.1

## Authors
**Deep Learning code:**<br>
Carlos Garcia, Keiichi Ito, Roman Feldbauer Javier Geijo and Wolfgang zu Castell

**Sample extraction code:**<br>
Nico Schreiber.<br>
**In process to be published. Available only upon request.**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


