#!/bin/bash

set -e

#ROOT=$(dirname $(readlink -e $0))
ROOT="$(pwd)"
LOCAL_MACHINE_SCRATCH_SPACE=/home/scratch
ENV_BASE="$(mktemp -u -d -p "$LOCAL_MACHINE_SCRATCH_SPACE" "conda_env.$USER.XXXXXXXXXXXX")"
ENV="$ENV_BASE/conda"
DATA="$ENV_BASE/data"

function safe_call {
    # usage:
    #   safe_call function param1 param2 ...

    HERE=$(pwd)
    "$@"
    cd "$HERE"
}

function conda_install {
    conda install --yes "$1"
}

function pip_install {
    pip install "$1"
}

function install_pylearn2 {
    if [ -d "pylearn2" ]; then
        echo "Existing version of pylearn2 found, removing."
        rm -rf pylearn2
    fi  

    git clone https://github.com/lisa-lab/pylearn2.git

    cd pylearn2
    python setup.py develop
}

function install_mnist {
    mkdir -p "$DATA/mnist"
    cd "$DATA/mnist"

    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

    gunzip *.gz
}

conda update --yes conda

conda create --yes --prefix "$ENV" python pip
source activate "$ENV"

# https://github.com/ContinuumIO/anaconda-issues/issues/32
cp .qt.conf "$ENV/bin/qt.conf"

echo "$ENV" > "$ROOT/.env"
echo "$DATA" > "$ROOT/.data"

safe_call conda_install numpy
safe_call conda_install scipy
safe_call conda_install matplotlib
safe_call conda_install pil
safe_call conda_install pyyaml
safe_call pip_install git+git://github.com/Theano/Theano.git
safe_call install_pylearn2
safe_call install_mnist
