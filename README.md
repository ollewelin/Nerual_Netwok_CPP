# Nerual_Netwok_CPP
Neural Network

Update with softmax
![](MNIST_with_softmax.png)

## Test residual_net.cpp
This network consist of 3 blocks, 3 fc_m_resnet object stacked on each other

    fc_m_resnet fc_nn_top_block;
    fc_m_resnet fc_nn_mid_block;
    fc_m_resnet fc_nn_end_block;

### Change Makefile to residual_net.cpp

    #SRCS = main.cpp fc_m_resnet.cpp simple_nn.cpp
    #PROG = main

    SRCS = residual_net.cpp fc_m_resnet.cpp 
    PROG = residual_net


 There are skip residual connection betwheen the input side of `fc_nn_end_block` and output side of `fc_nn_top_block` 
 to make a residual connection for not vanishing gradient esspecial if many mid blocks are stacked 
 
 The residual_net.cpp net seems to works and steady converge during training.
 The could be arbriatary size of input output nodes of all blocks even when use skip residual connection enabled
 `o_layer_delta`and `i_layer_delta` link the backpropagation between each `fc_m_resnet` object block 
 
 `use_skip_connect_mode = 1` enable skip residulal connections
 
 `block_type` set if the `fc_m_resnet` is a top, mid or end block. It is possible to stack many mid blocks with residual skip connections
 
 `fc_nn_end_block` dont have skip capability yet in version 0.0.6 will be added later 
 
 ## Here trained on MNIST Fashion dataset not the downloaded MNIST digits 
 
    olle@olle-TUF-Gaming-FX505DT-FX505DT:~/pytorch_cpp/t14/Nerual_Netwok_CPP$ ./residual_net 
    General Neural Network Residual net test Beta version under work...
    3 stackaed nn blocks with residual connections 
    file_size 47040016
    MNIST_file_size = 47040016
    train.. or t10k.. ..-images-idx3-ubyte file is successfully loaded in to MNIST_data[MN_index] memory
    file_size 60008
    train... or t10k...  ...-labels-idx1-ubyte file is successfully loaded in to MNIST_lable[MN_index] memory
    fc_m_resnet Constructor
    Seed radomizer done
    fc_m_resnet Constructor
    Seed radomizer done
    fc_m_resnet Constructor
    Seed radomizer done
    Fully connected residual neural network object
    fc_m_resnet object version : 0.0.6
    Number of hidden layers is set to = 1
    Size of hidden_layer[0][x] = 300
    hidden_layer vector is now set up
    Now setup all_weight, change_weights vectors size of this fc block
    Size of temporary dummy_1D_weight_vector from input layer connection[0] is = 785
    Size of temporary dummy_1D_weight_vector last hidden layer connection[0] is = 301
    The size of all_weight and change_weights in now setup OK !
    Note that the program how call this object could only set this size once. No protections against change size of the public vectors
    Setup state = 2
    Size of layer dimentions[] of weights at the this nn block = 2
    Size of node dimentions[][] of weights for hidden layer number 0 is: 300
    Size of weight dimentions[][][] of weights for hidden layer number 0 is: 785
    Size of node dimentions[][] of weights for hidden layer number 1 is: 30
    Size of weight dimentions[][][] of weights for hidden layer number 1 is: 301
    Number of hidden layers is set to = 3
    Size of hidden_layer[0][x] = 30
    Size of hidden_layer[1][x] = 50
    Size of hidden_layer[2][x] = 30
    hidden_layer vector is now set up
    Now setup all_weight, change_weights vectors size of this fc block
    Size of temporary dummy_1D_weight_vector from input layer connection[0] is = 31
    Size of temporary dummy_1D_weight_vector from hidden layer connection[1] is = 31
    Size of temporary dummy_1D_weight_vector from hidden layer connection[2] is = 51
    Size of temporary dummy_1D_weight_vector last hidden layer connection[2] is = 31
    The size of all_weight and change_weights in now setup OK !
    Note that the program how call this object could only set this size once. No protections against change size of the public vectors
    Setup state = 2
    Size of layer dimentions[] of weights at the this nn block = 4
    Size of node dimentions[][] of weights for hidden layer number 0 is: 30
    Size of weight dimentions[][][] of weights for hidden layer number 0 is: 31
    Size of node dimentions[][] of weights for hidden layer number 1 is: 50
    Size of weight dimentions[][][] of weights for hidden layer number 1 is: 31
    Size of node dimentions[][] of weights for hidden layer number 2 is: 30
    Size of weight dimentions[][][] of weights for hidden layer number 2 is: 51
    Size of node dimentions[][] of weights for hidden layer number 3 is: 30
    Size of weight dimentions[][][] of weights for hidden layer number 3 is: 31
    Number of hidden layers is set to = 1
    Size of hidden_layer[0][x] = 15
    hidden_layer vector is now set up
    Now setup all_weight, change_weights vectors size of this fc block
    Size of temporary dummy_1D_weight_vector from input layer connection[0] is = 31
    Size of temporary dummy_1D_weight_vector last hidden layer connection[0] is = 16
    The size of all_weight and change_weights in now setup OK !
    Note that the program how call this object could only set this size once. No protections against change size of the public vectors
    Setup state = 2
    Size of layer dimentions[] of weights at the this nn block = 2
    Size of node dimentions[][] of weights for hidden layer number 0 is: 15
    Size of weight dimentions[][][] of weights for hidden layer number 0 is: 31
    Size of node dimentions[][] of weights for hidden layer number 1 is: 10
    Size of weight dimentions[][][] of weights for hidden layer number 1 is: 16
    Do you want to load weights from saved weight file = Y/N 
    n
    Randomize weights 3D vector all weights of fc_resnet object....
    Randomize weights is DONE!
    setup_state = 3
    Randomize weights 3D vector all weights of fc_resnet object....
    Randomize weights is DONE!
    setup_state = 3
    Randomize weights 3D vector all weights of fc_resnet object....
    Randomize weights is DONE!
    setup_state = 3
    target_lable = 3
    Epoch ----0
    input node --- [0] = 0
    Epoch 0
    input node [0] = 1.01103
    Output node [0] = 0.0996975  Target node [0] = 1
    Output node [1] = 0.094681  Target node [1] = 0
    Output node [2] = 0.0968199  Target node [2] = 0
    Output node [3] = 0.108413  Target node [3] = 0
    Output node [4] = 0.0961141  Target node [4] = 0
    Output node [5] = 0.108008  Target node [5] = 0
    Output node [6] = 0.100616  Target node [6] = 0
    Output node [7] = 0.0916473  Target node [7] = 0
    Output node [8] = 0.0931175  Target node [8] = 0
    Output node [9] = 0.110886  Target node [9] = 0
    Training loss = 138426
    correct_classify_cnt = 5998
    correct_ratio = 9.99667
    Epoch ----1
    input node --- [0] = 1.01103
    Epoch 1
 
    ...
    ... 
 
    Training loss = 21671.6
    correct_classify_cnt = 52173
    correct_ratio = 86.955
    Epoch ----19
    input node --- [0] = 0.795827
    Epoch 19
    input node [0] = 0.151538
    Output node [0] = 0.0851303  Target node [0] = 0
    Output node [1] = 0.00915264  Target node [1] = 0
    Output node [2] = 0.691792  Target node [2] = 1
    Output node [3] = 0.0190685  Target node [3] = 0
    Output node [4] = 0.120822  Target node [4] = 0
    Output node [5] = 6.02811e-05  Target node [5] = 0
    Output node [6] = 0.0738069  Target node [6] = 0
    Output node [7] = 3.25776e-05  Target node [7] = 0
    Output node [8] = 7.37132e-05  Target node [8] = 0
    Output node [9] = 6.13079e-05  Target node [9] = 0
    Training loss = 21475.6
    correct_classify_cnt = 52252
    correct_ratio = 87.0867
    Save data weights ...
    Save data finnish !
    Save data weights ...
    Save data finnish !
    Save data weights ...
    Save data finnish !
    Epoch ----20

![](residual_net_7_layer_in_total_c.png)

