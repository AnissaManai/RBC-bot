# FROM ubuntu:latest
FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN apt-get update && apt-get install -y \
	wget \
    git

#Install MINICONDA
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/miniconda3 && \
	rm Miniconda.sh

ENV PATH /opt/miniconda3/bin:$PATH

# clone repository
WORKDIR /
RUN git clone https://github.com/AnissaManai/RBC-bot.git

WORKDIR /RBC-bot

# Create conda Environment usinng yml file
RUN conda env create -f environment.yml python=3.6 && \
	/opt/miniconda3/envs/rbc/bin/pip install --user jupyterlab 
	
ARG BUILD_TIMESTAMMP=unknown



WORKDIR /RBC-bot
RUN git pull

SHELL ["conda", "run", "--no-capture-output", "-n", "rbc", "/bin/bash", "-c"]

EXPOSE 8888

# CMD /opt/miniconda3/envs/rbc/bin/jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port=8888


CMD ["conda", "run", "--no-capture-output", "-n", "rbc", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]




