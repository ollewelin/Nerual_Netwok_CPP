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

