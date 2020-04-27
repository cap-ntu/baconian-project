FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    libpcre16-3 \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN wget https://www.roboti.us/download/mjpro131_linux.zip
RUN wget https://www.roboti.us/download/mjpro150_linux.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip
RUN unzip mjpro131_linux.zip -d /root/.mujoco/
RUN unzip mjpro150_linux.zip -d /root/.mujoco/
RUN unzip mujoco200_linux.zip -d /root/.mujoco/

RUN mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200

RUN rm mjpro131_linux.zip && rm mjpro150_linux.zip && rm mujoco200_linux.zip


COPY . /baconian-project
COPY ./mjkey.txt /root/.mujoco/

# ENV LD_LIBRARY_PATH /root/.mujoco/mjpro131/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
# ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN source ~/.bashrc

WORKDIR /baconian-project
RUN pip install cffi && pip install pip -U && pip install -e .
ENTRYPOINT ["python", "/baconian-project/benchmark/run_benchmark.py"]