
CC = g++ -std=c++14 -O3
CFLAGS = -g -Wall

#SRCS = main.cpp fc_m_resnet.cpp simple_nn.cpp
#PROG = main

#SRCS = convolution_nn.cpp fc_m_resnet.cpp load_mnist_dataset.cpp convolution.cpp
#PROG = convolution_nn

SRCS = residual_net.cpp fc_m_resnet.cpp load_mnist_dataset.cpp convolution.cpp
PROG = residual_net

#SRCS = verify.cpp fc_m_resnet.cpp simple_nn.cpp
#PROG = verify

#OPENCV = `pkg-config opencv --cflags --libs`
#LIBS = $(OPENCV)
LIBS =

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
