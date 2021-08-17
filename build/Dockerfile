ARG OWNER=jupyter
ARG BASE_CONTAINER=$OWNER/scipy-notebook
FROM $BASE_CONTAINER

#LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER root

RUN apt-get update --yes
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install -U pip

USER ${NB_UID}

# Install Tensorflow
RUN mamba install --quiet --yes \
    'tensorflow=2.4.1' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"