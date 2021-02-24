# Python 3.6 installation (Ubuntu)
The following will gide you to install Python 3.6 in an Ubuntu system, however, you can replicate in most linux systems and for different python versions.

1. Update the repositories
`sudo apt update`
2. Install dependencies
`sudo apt install wget gcc g++ gfortran build-essential libc6-dev libbz2-dev software-properties-common make`
3. Download the source code of Python 3.6.X and untar the zip file
`wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tar.xz`
`tar -xvf Python-3.6.8.tar.xz`
4. Change to folder Python-3.6.8 and run the configure file
`cd Python-3.6.8`
`./configure --enable-shared`
5. Install with make
`make altinstall`
6. Check Python version.
`python --version` or `python3 --version`
