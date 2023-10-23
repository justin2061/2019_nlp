ARG OWNER=jupyter
ARG BASE_CONTAINER=$OWNER/tensorflow-notebook
FROM $BASE_CONTAINER
ARG DEBIAN_FRONTEND=noninteractive

#LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER root

# Workaround https://unix.stackexchange.com/questions/2544/how-to-work-around-release-file-expired-problem-on-a-local-mirror
RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until
RUN apt-get update --yes
RUN apt-get -y install apt-utils
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install -U pip
# https://github.com/jupyter-server/jupyter-resource-usage
RUN pip install jupyter-resource-usage

USER ${NB_UID}

#RUN pip install tensorflow==2.12.0
RUN pip install torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers
RUN pip install -q nltk