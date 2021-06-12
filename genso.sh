#!/bin/bash

gcc rec2buf.c -fPIC -W -Wall -O3 -fopenmp --shared -o libio.so
#clang rec2buf.c -fPIC -W -Wall -O3 -fopenmp  --shared -o libio.so

