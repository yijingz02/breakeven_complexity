#!/bin/bash


docker run -it --rm \
	--gpus '"device=0,1"' \
  -v $(pwd):/workspace \
  pyfr:latest /bin/bash
