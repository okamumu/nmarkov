FROM jupyter/datascience-notebook

USER root
RUN apt-get -y update &&\
        apt-get install -y \
            libc-bin

USER jovyan
RUN conda install -y numpy scipy matplotlib pybind11
RUN pip install git+https://github.com/okamumu/nmarkov.git
