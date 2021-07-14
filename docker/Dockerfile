FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN apt update -y && apt install xauth libgl1-mesa-glx -y && touch ~/.Xauthority
RUN conda install pillow -y && conda install libwebp -y && conda install -c conda-forge libwebp -y && conda install pillow -y
