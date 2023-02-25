#include "convolution.hpp"
#include <iostream>
#include <fstream>

using namespace std;

convolution::convolution()
{
    version_major = 0;
    version_mid = 0;
    version_minor = 1;
    //0.0.1 Almost empty class no algorithm yet implemented
    setup_state = 0;
    kernel_size = 3;
    stride = 0;
    cout << "Constructor Convloution neural network object " << endl;
}

convolution::~convolution()
{
    cout << "Destructor Convloution neural network object " << endl;
}

void convolution::set_kernel_size(int k_size)
{
    if(setup_state > 1)
    {
        cout << "Error could not change kernel when set_out_tensor_channels() already setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }

    if(k_size < 1)
    {
        cout << "Error kernel size is set < 1, k_size = " << k_size << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }

    int check_odd = k_size % 2;
    if(check_odd == 0)
    {
        cout << "Error kernel size is even. kernel size must always be an odd number. k_size = " << k_size << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }
    else
    {
        cout << "OK kernel size = " << k_size << endl; 
    }
    kernel_size = k_size;
    setup_state = 1;
}
int convolution::get_kernel_size()
{

    return kernel_size;
}
void convolution::set_stride(int stride_size)
{
    if(setup_state > 1)
    {
        cout << "Error could not change stride size when set_out_tensor_channels() already setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }
    if(stride_size > 1 || stride_size < 0)
    {
        cout << "Error stride size out of range. stride = " << stride_size << endl; 
        cout << "Stride size on convolution version " << ver_major << "." << version_mid << "." << version_minor << " is only support range 0-1" << endl;
        cout << "Exit program" << endl; 
        exit(0);
    }
    cout << "OK stride size is set to " << stride_size << endl; 
    stride = stride_size;
}
int convolution::get_stride()
{
    return stride;
}

void convolution::set_out_tensor_channels(int out_depth)
{
    if(setup_state != 1)
    {
        cout << "Error could not set_out_tensor_channels() setup_state must be = 1 when call set_out_tensor_channels(), setup_state = " << setup_state << endl; 
        cout << "setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }

    

    output_tensor_channels = out_depth;
    //TODO...
    setup_state = 2;
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