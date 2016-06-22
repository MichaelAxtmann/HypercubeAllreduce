FLAGS=-std=c++11 -Wall -Wextra
CXX=mpic++
all: release

debug:
	$(CXX) $(FLAGS) -O0 -ggdb -o allreduce HypercubeAllreduce.cpp

release:
	$(CXX) $(FLAGS) -O3 -g -DNDEBUG -o allreduce HypercubeAllreduce.cpp
