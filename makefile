all:
	g++ -march=native -O3 -std=gnu++11 -ffast-math -pipe -larmadillo -lboost_iostreams main.cc -o main