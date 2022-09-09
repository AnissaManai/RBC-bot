FROM nvcr.io/nvidia/pytorch:21.06-py3
# FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
	gcc \
	g++ \
	curl \
	wget \
    git \
    make \
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
RUN conda env create -f environment.yml python=3.6
	
WORKDIR /
#### Compile stockfish. outputs to /Stockfish/src/stockfish ####
RUN wget https://www.dropbox.com/sh/75gzfgu7qo94pvh/AAAtWIKGvQvGih-dz6xb-CHDa/Stockfish%2011/stockfish-11-linux.zip?dl=0 && \
	unzip stockfish-11-linux.zip?dl=0 && \
	cd /stockfish-11-linux/Linux && \
	chmod +x stockfish_20011801_x64_bmi

ENV STOCKFISH_EXECUTABLE /stockfish-11-linux/Linux/stockfish_20011801_x64_bmi

ARG BUILD_TIMESTAMMP=unknown

WORKDIR /RBC-bot
RUN git pull

SHELL ["conda", "run", "--no-capture-output", "-n", "rbc", "/bin/bash", "-c"]

EXPOSE 8888

# CMD /opt/miniconda3/envs/rbc/bin/jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port=8888


CMD ["conda", "run", "--no-capture-output", "-n", "rbc", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]




