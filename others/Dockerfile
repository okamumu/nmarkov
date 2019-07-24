FROM jupyter/datascience-notebook

USER root
RUN apt-get -y update &&\
        apt-get install -y \
            libc-bin \
            libopenblas-base \
            antlr4

USER jovyan
RUN conda install -c anaconda openblas
RUN conda install -c conda-forge pydotplus
RUN pip install z3-solver pybind11 antlr4-python3-runtime
