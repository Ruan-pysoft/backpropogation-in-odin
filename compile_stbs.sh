#!/bin/sh

gcc -c -x c -DSTB_IMAGE_IMPLEMENTATION stb_image.h
gcc -c -x c -DSTB_IMAGE_WRITE_IMPLEMENTATION stb_image_write.h

ar rcs stb_image.a stb_image.o stb_image_write.o

rm stb_image.o
rm stb_image_write.o
