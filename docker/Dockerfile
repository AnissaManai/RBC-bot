# FROM ubuntu:latest
FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN apt-get update && apt-get install -y \
	wget \
    git

#Install MINICONDA
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# clone repository
WORKDIR /
RUN git clone https://github.com/AnissaManai/RBC-bot.git

WORKDIR /RBC-bot

# Create conda Environment usinng yml file
RUN conda env create -f environment.yml python=3.6 && \
	/opt/conda/envs/rbc/bin/pip install jupyterlab

CMD /opt/conda/envs/rbc/bin/jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port=8888


# CMD ["conda", "run", "-n", "rbc", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]



