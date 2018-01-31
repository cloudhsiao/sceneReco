# Run CTPN with python 3.6 in macOS 10.13 (NO GPU supported)

### step 1. Install dependencies

Generally you can following this instruction to install prerequsities (http://caffe.berkeleyvision.org/install_osx.html).
But we should finetune some steps to install modules for python3

> Note: Just install dependencies of caffe, we will not follow the instructions to compile caffe

```
$ brew install snappy leveldb gflags glog szip lmdb python3
$ brew brew tap homebrew/science
$ brew install hdf5 opencv

$ brew install --build-from-source boost
$ brew install --build-from-source --with-python3 boost-python
$ brew install --build-from-source --with-python3 protobuf
```

### step 2. compile the caffe

I can not compile with atlas(vecLib), so I use openblas here.

```
$ brew uninstall openblas; brew install --fresh -vd openblas
```

You may follow my Makefile.config or finetune this file to fit your system.

```
$ cd caffe
$ make clean; make pycaffe
$ cd ..
```

### step 3. test

```
$ python3 tools/demo.py --no-gpu

...ignore...

input exit break


please input file name:21.bmp
tcmalloc: large alloc 1244160000 bytes == 0x140ae6000 @  0x7fff673be201 0x7fff673bd50b 0x110629d09 0x110629aec 0x110629c7e 0x11057964f 0x11056e1ba 0x1105f9dae 0x110603751 0x110388e63 0x110dfdfa6 0x110e008ea 0x110e067be 0x110e00423 0x106f51cb7 0x106ff297d 0x106feb317 0x106ff3155 0x106ff389b 0x106ff2984 0x106feb317 0x106ff3b43 0x106ff2984 0x106feb317 0x106ff3b43 0x106ff2984 0x106feb317 0x106ff3b43 0x106ff2984 0x106feb317 0x106ff3155
Number of the detected text lines: 1
Time: 8.102464
```

Done!
