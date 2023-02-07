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
 
 The net works seems to stady converge  during training but slowly because of deep structure
 
 `fc_nn_end_bloc` dont have skip capability yet in version 0.0.6 will be added later 

![](residual_net_7_layer_in_total_c.png)

