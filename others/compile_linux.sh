g++ -std=c++11 -Wall -shared -fPIC \
  -I pybind11/include -I marlib \
  `python3-config --cflags --ldflags` \
  nmarkov/src/nmarkov.cpp \
  -o nmarkov/_nmarkov`python3-config --extension-suffix` \
  -l blas
