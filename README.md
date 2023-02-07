# Nerual_Netwok_CPP
Neural Network

Update with softmax
![](MNIST_with_softmax.png)

## Test residual_net.cpp
This network consist of 3 blocks och 3 fc_m_resnet object tacked on each other

    fc_m_resnet fc_nn_top_block;
    fc_m_resnet fc_nn_mid_block;
    fc_m_resnet fc_nn_end_block;
    
 There are skip residual connection betwheen the input side of `fc_nn_end_block` and output side of `fc_nn_top_block` 
 to make a residual connection for not vanishing gradient esspesialy if many mid blocks are stacked 
 
 The net seems to works and stady converge during training but slowly because of deep structure
 The could be arbriatary size of input output nodes of all blocks even when use skip residual connection enabled
 `o_layer_delta`and `i_layer_delta` link the backpropagation betwheen each `fc_m_resnet` object block 
 
 `use_skip_connect_mode = 1` enable skip residulal connections
 
 `fc_nn_end_bloc` dont have skip capability yet in version 0.0.6 will be added later 

![](residual_net_7_layer_in_total_c.png)

