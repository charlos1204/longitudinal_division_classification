# Python 3.6 installation (Ubuntu)
The following will gide you to install Python 3.6 in an Ubuntu system, however, you can replicate in most linux systems and for different python versions.

1. Update the repositories<br>
`sudo apt update`
2. Install dependencies<br>
`sudo apt install wget gcc g++ gfortran build-essential libc6-dev libbz2-dev software-properties-common make`
3. Download the source code of Python 3.6.X and untar the zip file<br>
`wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tar.xz`<br>
`tar -xvf Python-3.6.8.tar.xz`
4. Change to folder Python-3.6.8 and run the configure file<br>
`cd Python-3.6.8`<br>
`./configure --enable-shared`
5. Install with make<br>
`make altinstall`
6. Check Python version.<br>
`python --version`<br> 
or<br>
`python3 --version`
