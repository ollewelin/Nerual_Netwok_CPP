
CC = g++ -std=c++14 -O3
CFLAGS = -g -Wall
SRCS = main.cpp fc_m_resnet.cpp
PROG = main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)