docker build -t jupyter/z3 .
docker run -it --name notebook --rm -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/z3
