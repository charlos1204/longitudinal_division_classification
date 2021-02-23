# Efficient detection of longitudinal bacteria fission using transfer learning in Deep Neural Networks

One Paragraph of project description goes here

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites
* git (optional, https://git-scm.com/)
* Nvidia driver (https://www.nvidia.com)
* cuda version 10 or higher (https://developer.nvidia.com/cuda-10.0-download-archive)
* Docker (optional, https://docs.docker.com/get-docker/)
* GPU

The following libraries can be install with miniconda or pip.
* miniconda (optional, https://docs.conda.io/en/latest/miniconda.html)
* python 3.6 or higher (https://www.python.org/)
* cython <-----------
* opencv <-----------
* tqdm <-----------
* torchsummary <----------
* matplotlib <---------
* scipy <---------
* pytorch <----------
* JupyterLab (https://jupyter.org/)
* torch, torchvision (https://pytorch.org/get-started/locally/)
* sklearn (https://scikit-learn.org/stable/index.html)
* numpy (https://numpy.org/)
* pandas (https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html)

## Installation
#### 1. Git for cloning the project (optional)
* To install git in Debian based distribution (Ubuntu):
`
$ sudo apt install git-all
` 
* To install git in Windows download the binary from [here](https://git-scm.com/download/win)
* To install git in macOS download the binary from [here](https://git-scm.com/download/mac)

#### 2. Nvidia driver and cuda for the GPU card
* To install Nvidia driver for Linux, macOS, or Widnows will depend on the card you have, you can check the steps [here](https://www.nvidia.com/Download/index.aspx?lang=en-us), and for the cuda library [here](https://developer.nvidia.com/cuda-10.0-download-archive), we recommend asking your IT administrator for help.

#### 3. Docker for running the container (optional)
Only for running the Docker image of the code.
* to install Docker in Debian based distribution (Ubuntu):
`
$ sudo apt update
`
`
$ sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
`
`
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
`
`
$ sudo apt-key fingerprint 0EBFCD88
`
`
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
`
`
$ sudo apt-get update
`
`
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
`
* To install Docker in Windows follow the steps [here](https://docs.docker.com/docker-for-windows/install/)
* To install Docker in macOS follow the steps [here](https://docs.docker.com/docker-for-mac/install/)

Run this example in a terminal to make sure Docker is installed correctly:
`
$ sudo docker run hello-world
`

## Deep learning libraries
The easiest way to install python and the libraries to run the code is through miniconda. Nevertheles, linux distributions already have python and pip installed. Pip is the package installer for Python. If you prefer try with miniconda, follow the step 4. Otherwise skip to step 5 to use pip.

#### 4. Install miniconda
* Download the installer from [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers) depending on your operating system.
* For linux run the following command:
`
bash Miniconda3-latest-Linux-x86_64.sh
`
* For Windows: double-click the .exe file and follow the instructions on the screen.
* For macOS run the following command:
`
bash Miniconda3-latest-MacOSX-x86_64.sh
`

Install libraries to run the code.
```
conda install -c anaconda cython
conda install -c conda-forge opencv
conda install -c conda-forge tqdm
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-model-summary 
conda install -c conda-forge matplotlib
conda install -c anaconda scipy
conda install -c anaconda pandas
conda install -c anaconda joblib
conda install -c anaconda scikit-learn
conda install -c conda-forge jupyterlab
```
