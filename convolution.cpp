#include "convolution.hpp"
#include <iostream>
#include <fstream>

using namespace std;

convolution::convolution()
{
    version_major = 0;
    version_mid = 0;
    version_minor = 1;

    setup_state = 0;
    cout << "Constructor Convloution neural network object " << endl;
}

convolution::~convolution()
{
    cout << "Destructor Convloution neural network object " << endl;
}

void convolution::set_kernel_size(int k_size)
{
    kernel_size = k_size;
}
int convolution::get_kernel_size()
{
    return kernel_size;
}
void convolution::set_out_tensor_channels(int out_depth)
{
    output_tensor_channels = out_depth;
}
void convolution::randomize_weights(double rand_prop)
{
    
}
void convolution::load_weights(string filename)
{
    
}
void convolution::save_weights(string filename)
{
    
}
void convolution::conv_forward()
{
    
}
void convolution::conv_backprop()
{
    
}
void convolution::conv_update_weights()
{
    
}
void convolution::conv_transpose_fwd()
{
    
}
void convolution::conv_transpose_bkp()
{
    
}

void convolution::get_version()
{
    cout << "Convloution neural network object" << endl;
    cout << "Convloution version : " << version_major << "." << version_mid << "." << version_minor << endl;
    ver_major = version_major;
    ver_mid = version_mid;
    ver_minor = version_minor;
}