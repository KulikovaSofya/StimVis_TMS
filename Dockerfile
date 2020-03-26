FROM ubuntu:18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

MAINTAINER Sofya KULIKOVA <spkulikova@hse.ru>

RUN apt-get update
RUN apt-get install -y mesa-utils
RUN apt-get install -y libxt-dev

RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y xvfb
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libfontconfig1

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && mkdir /root/.conda/ \
	&& bash Miniconda3-latest-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-latest-Linux-x86_64.sh 

# Setting and installing environment
RUN wget https://github.com/simnibs/simnibs/releases/latest/download/environment_linux.yml
RUN conda env create --file env_linux.yml 

SHELL ["conda", "run", "-n", "simnibs_env", "/bin/bash", "-c"]

RUN pip install pytest
RUN pip install -f https://github.com/simnibs/simnibs/releases/latest simnibs
RUN mkdir $HOME/SimNIBS && postinstall_simnibs -s --copy-matlab --setup-links -d $HOME/SimNIBS

RUN pip install matplotlib
RUN pip install dipy
RUN pip install vtkplotter
RUN pip install fury

COPY tms_stimulation.py ./

ENTRYPOINT ["conda", "run", "-n", "simnibs_env", "python", "tms_stimulation.py" ]
