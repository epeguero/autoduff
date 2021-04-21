# FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y apt-utils
# RUN debconf-set-selections "debconf debconf/frontend select Noninteractive"
RUN apt-get install -y git python3 python3-pip python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev 

WORKDIR /workspace/
RUN git clone --recursive https://github.com/apache/incubator-tvm

WORKDIR /workspace/incubator-tvm

RUN apt-get install -y ninja-build
RUN pip3 install --user numpy decorator attrs tornado 

RUN apt-get install -y wget lsb-release software-properties-common
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

RUN mkdir build && cp cmake/config.cmake build
RUN sed -i 's/USE_CUDA OFF/USE_CUDA ON/g' build/config.cmake
RUN sed -i 's/USE_LLVM OFF/USE_LLVM ON/g' build/config.cmake
RUN cd build && cmake .. -G Ninja && ninja

RUN cd python; python3 setup.py install --user; cd ..


WORKDIR /workspace

# RUN export TVM_HOME=/usr/tvm
# RUN export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.1/compat:${LD_LIBRARY_PATH}

# Dependencies for TVM auto-tuning
# cmake in apt-get is too old
RUN apt remove -y cmake
RUN mkdir cmake
WORKDIR /workspace/cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.0-rc2/cmake-3.19.0-rc2-Linux-x86_64.sh
RUN chmod +x cmake-3.19.0-rc2-Linux-x86_64.sh \
    && bash cmake-3.19.0-rc2-Linux-x86_64.sh --skip-license
ENV PATH="/workspace/cmake/bin:${PATH}"

# RUN cd /usr/local/bin \
#     && ln -s /workspace/cmake-3.19.0-rc2-Linux-x86_64/bin cmake \
#     && ls

WORKDIR /workspace

# RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.0-rc2/cmake-3.19.0-rc2-Linux-x86_64.tar.gz
# RUN apt remove -y cmake \
#     && tar -xvf cmake-3.19.0-rc2-Linux-x86_64.tar.gz \
#     && cd cmake-3.19.0-rc2-Linux-x86_64  \
#     && ls \
#     && ./configure \
#     && make \
#     && make install
    
     
RUN pip3 install --user psutil xgboost tornado
# RUN pip3 install --user cython
# RUN find / -type d -name tvm
# RUN cd /root/.local/lib/python3.6/site-packages/tvm-0.8.dev120+g0c7aae345-py3.6-linux-x86_64.egg/tvm && make cython3
 

# Python Versions
# RUN pip3 install torch==1.4.0 torchvision==0.5.0
RUN pip3 install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install matplotlib colorlog pytest bitstring 
# ENV TZ=America/New_York
# RUN echo "tzdata tzdata/Areas select America" > preseed.txt
# RUN echo "tzdata tzdata/Zones/America select New_York" >> preseed.txt
# RUN debconf-set-selections preseed.txt
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
# RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install python3-tk
RUN apt-get install -y --no-install-recommends python3-tk

ENTRYPOINT python3 src/fuzz.py
