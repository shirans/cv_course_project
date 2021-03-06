#!/usr/bin/env bash

# cd cv_course_project
# chmod +x install.sh so it'll be runnable
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7
sudo apt-get install -y libjpeg8-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
sudo apt-get build-dep -y python-imaging
sudo apt-get install -y libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
PYTHON_VERSIONS="3.7"
sudo apt-get install -y python-dev python3-dev
# sudo apt-get install -y python-pip python3-pip
wget https://bootstrap.pypa.io/get-pip.py && \
    for VERSION in ${PYTHON_VERSIONS}; do sudo python$VERSION get-pip.py; done && \
    rm get-pip.py
for VERSION in ${PYTHON_VERSIONS}; do pip$VERSION install --upgrade pip; done
sudo pip3.7 install virtualenv
virtualenv .env

source .env/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install jupyter

jupyter notebook --generate-config

pip3 install -r requirements.txt

#deactivate

# when editing ~/.jupyter/jupyter_notebook_config.py
#c = get_config()
#c.NotebookApp.ip = '0.0.0.0'
#c.NotebookApp.open_browser = False
#c.NotebookApp.port = 8888