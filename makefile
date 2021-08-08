build: $(wildcard src/*) ./makefile
	mkdir -p build ; cd build ; cmake .. -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=g++; make -j 12

release: $(wildcard src/*) ./makefile
	mkdir -p build ; cd build ; cmake .. -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ ; make -j 12

clean:
	rm -rf build
