## commands to build and run this file
# docker build -t deep-conv-rf:latest - < Dockerfile
# docker run -it --rm --name deep-conv-rf-env deep-conv-rf:latest

FROM ubuntu:16.04

# set maintainer
LABEL maintainer="spalani2@jhu.edu"

# update
RUN apt-get update && apt-get -y upgrade

# install packages
RUN apt-get install -y \
    cmake \
    cpio \
    gfortran \
    libpng-dev \
    freetype* \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    software-properties-common\
    git \
    man \
    wget

# install python3
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y \
  python3.6 \
  python3.6-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

RUN ln -s /usr/bin/python3.6 /usr/local/bin/python

# Install Rerf dependencies
RUN apt-get install -y build-essential cmake python3-dev libomp-dev vim

# make a directory for mounting local files into docker
RUN mkdir /root/host_files/

# change working directory to install RerF
RUN mkdir /root/code/
WORKDIR /root/code/

# clone the RerF code into the container
RUN git clone https://github.com/neurodata/RerF.git .

# go to Python subdir (install python bindings)
WORKDIR /root/code/Python

# install python requirements
RUN pip install -r requirements.txt
RUN pip install matplotlib seaborn pandas jupyter pycodestyle torch torchvision

# clean old installs
RUN python setup.py clean --all

# install RerF
RUN pip install -e .

# add RerF to PYTHONPATH for dev purposes
RUN echo "export PYTHONPATH='${PYTHONPATH}:/root/code'" >> ~/.bashrc

# clean dir and test if mgcpy is correctly installed
RUN py3clean .
RUN python -c "import RerF"

# go back to code root
WORKDIR /root/code

# launch terminal
CMD ["/bin/bash"]
