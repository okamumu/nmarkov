# clang++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
#     -fPIC `python -m pybind11 --includes` -I marlib -I include \
#     src/pymarkov.cpp -o pymarkov`python-config --extension-suffix`
