#!/bin/bash

docker run -it --rm \
        --gpus '"device=0"' \
  -v $(pwd):/workspace \
 eddyformer:latest /bin/bash

