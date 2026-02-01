mkdir -p build
cd build
# cmake -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ ..
cmake -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=g++ ..
make easy_gpt -j 8
