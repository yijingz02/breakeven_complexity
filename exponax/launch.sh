#!/bin/bash


docker run -it --rm \
	--gpus '"device=1"' \
  -v $(pwd):/workspace \
  exponax:latest /bin/bash
