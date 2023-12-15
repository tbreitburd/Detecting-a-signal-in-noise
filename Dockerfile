FROM continuumio/miniconda3

RUN mkdir -p S1_Coursework

COPY . /S1_Coursework

WORKDIR /S1_Coursework

RUN conda env update -f environment.yml --name PrincDSCW

RUN echo "conda activate PrincDSCW" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
