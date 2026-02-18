FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y python3 python3-pip libopenmpi-dev openmpi-bin git build-essential cmake

WORKDIR /app
RUN git clone https://github.com/libxsmm/libxsmm.git
WORKDIR /app/libxsmm
RUN make -j4 STATIC=0 BLAS=0
RUN export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.so

RUN pip install pyfr scipy

RUN apt-get -y install libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
RUN pip install gmsh shapely

RUN conda install -c conda-forge paraview

WORKDIR /workspace
