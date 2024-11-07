#!/bin/bash

cmake CMakeLists.txt
make 
mv fdapde_test build/ 
cd build/
./fdapde_test
rm fdapde_test
cd ../ 
