FROM continuumio/miniconda3

RUN mkdir -p S1_Coursework

COPY . /S1_Coursework

RUN cd S1_Coursework \
    && conda env update -f environment.yml --name PrincDSCW
