FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y python3 python3-pip libopenmpi-dev openmpi-bin git build-essential cmake

RUN pip install pyfr scipy
RUN pip install -U "jax[cuda12]"
RUN pip install phiflow the_well exponax h5py hdf5storage==0.2.0
RUN pip install --force-reinstall nvidia-cudnn-cu12==9.8.0.87

WORKDIR /workspace
