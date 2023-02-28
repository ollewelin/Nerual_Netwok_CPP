#include "convolution.hpp"
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

convolution::convolution()
{
    version_major = 0;
    version_mid = 0;
    version_minor = 0;
    // 0.0.0 Not finnish at all
    setup_state = 0;
    kernel_size = 3;
    stride = 1;
    dropout_proportion = 0.0;
    use_dopouts = 0;
    cout << "Constructor Convloution neural network object " << endl;
}

convolution::~convolution()
{
    cout << "Destructor Convloution neural network object " << endl;
}
void convolution::set_kernel_size(int k_size)
{
    if (setup_state > 1)
    {
        cout << "Error could not change kernel when set_out_tensor_channels() already setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    if (k_size < 1)
    {
        cout << "Error kernel size is set < 1, k_size = " << k_size << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    int check_odd = k_size % 2;
    if (check_odd == 0)
    {
        cout << "Error kernel size is even. kernel size must always be an odd number. k_size = " << k_size << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    else
    {
        cout << "   OK kernel size = " << k_size << endl;
    }
    input_side_size = input_tensor[0][0].size();
    if (input_side_size < k_size)
    {
        cout << "Error kernel size is > input side of square size input_side_size = " << input_side_size << endl;
        cout << "kernel_size = " << k_size << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    else
    {
        cout << "   OK input_side_size = " << input_side_size << endl;
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
    if (setup_state > 1)
    {
        cout << "Error could not change stride size when set_out_tensor_channels() already setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    if (stride_size > 2 || stride_size < 1)
    {
        cout << "   Error stride size out of range. stride = " << stride_size << endl;
        cout << "   Stride size on convolution version " << ver_major << "." << version_mid << "." << version_minor << " is only support range 1-2" << endl;
        cout << "   Exit program" << endl;
        exit(0);
    }
    cout << "   OK stride size is set to " << stride_size << endl;
    stride = stride_size;
}
int convolution::get_stride()
{
    return stride;
}

void convolution::set_in_tensor(int total_input_size_one_channel, int in_channels)
{
    input_tensor_channels = in_channels;

    if (setup_state != 0)
    {
        cout << "Error could not set_in_tensor() setup_state must be = 0 when call set_in_tensor, setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    cout << "   ========================================" << endl;
    cout << "   data_size_one_sample one channel = " << total_input_size_one_channel << endl;
    root_of_intdata_size = sqrt(total_input_size_one_channel);
    // cout << "   root_of_intdata_size = " << root_of_intdata_size << endl;
    int test_square_the_root = root_of_intdata_size * root_of_intdata_size;
    if (test_square_the_root != total_input_size_one_channel)
    {
        cout << "Error Indata one sample not able to make a square root without decimal, test_square_the_root = " << test_square_the_root << endl;
        cout << "Indata don't fit a perfect square, Exit program " << endl;
        exit(0);
    }

    vector<double> dummy_conv_1D_vector;
    vector<vector<double>> dummy_conv_2D_vector;
    //========= Set up convolution input tensor size for convolution object =================
    for (int i = 0; i < root_of_intdata_size; i++)
    {
        dummy_conv_1D_vector.push_back(0.0);
    }
    for (int i = 0; i < root_of_intdata_size; i++)
    {
        dummy_conv_2D_vector.push_back(dummy_conv_1D_vector);
    }
    for (int i = 0; i < input_tensor_channels; i++)
    {
        input_tensor.push_back(dummy_conv_2D_vector);
        i_tensor_delta.push_back(dummy_conv_2D_vector);
    }
}

void convolution::set_out_tensor(int out_channels)
{
    output_tensor_channels = out_channels;
    if (setup_state != 1)
    {
        cout << "Error could not set_out_tensor() setup_state must be = 1 when call set_out_tensor_channels(), setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    //========= Set up convolution weight tensor and output tensor size for convolution object =================
    if (stride == 1)
    {
        output_side_size = (input_side_size - kernel_size) + 1;
    }
    else if (stride == 2)
    {
        output_side_size = ((input_side_size - kernel_size) / 2) + 1;
    }
    cout << "   OK output_side_size = " << output_side_size << endl;
    // output_side_size = xxx
    vector<double> dummy_1D_vect;
    vector<vector<double>> dummy_2D_vect;
    vector<vector<vector<double>>> dummy_3D_vect;
    for (int i = 0; i < kernel_size; i++)
    {
        dummy_1D_vect.push_back(0.0);
    }
    for (int i = 0; i < kernel_size; i++)
    {
        dummy_2D_vect.push_back(dummy_1D_vect);
    }
    for (int i = 0; i < input_tensor_channels; i++)
    {
        dummy_3D_vect.push_back(dummy_2D_vect);
    }

    for (int i = 0; i < output_tensor_channels; i++)
    {
        kernel_weights.push_back(dummy_3D_vect); // 4D [output_channel][input_channel][kernel_row][kernel_col]
        change_weights.push_back(dummy_3D_vect); // 4D [output_channel][input_channel][kernel_row][kernel_col]
        kernel_deltas.push_back(dummy_3D_vect);  // 4D [output_channel][input_channel][kernel_row][kernel_col]
    }

    // Add also one bias weight at end for the hole [input_channel][0][0]
    for (int i = 0; i < output_tensor_channels; i++)
    {
        accum_bias_deltas.push_back(0.0);   //
        kernel_bias_weights.push_back(0.0); // kernel bias weight [output_channel]
        change_bias_weights.push_back(0.0); // change bias weight [output_channel]
        // kernel_deltas dont need space for bias. no bias here
    }

    dummy_1D_vect.clear();
    dummy_2D_vect.clear();
    for (int i = 0; i < output_side_size; i++)
    {
        dummy_1D_vect.push_back(0.0);
    }
    for (int i = 0; i < output_side_size; i++)
    {
        dummy_2D_vect.push_back(dummy_1D_vect);
    }
    for (int i = 0; i < output_tensor_channels; i++)
    {
        output_tensor.push_back(dummy_2D_vect);
        o_tensor_delta.push_back(dummy_2D_vect);
        internal_tensor_delta.push_back(dummy_2D_vect); // o_tensor_delta derivated backwards to inside the activation fucntion
    }

    cout << "   kernel_bias_weights.size() = " << kernel_bias_weights.size() << endl;
    cout << "   kernel_weights.size() = " << kernel_weights.size() << endl;
    cout << "   kernel_weights[0].size() = " << kernel_weights[0].size() << endl;
    cout << "   kernel_weights[0][0].size() = " << kernel_weights[0][0].size() << endl;
    cout << "   kernel_weights[" << output_tensor_channels - 1 << "][" << input_tensor_channels - 1 << "][" << kernel_size - 1 << "].size() = " << kernel_weights[output_tensor_channels - 1][input_tensor_channels - 1][kernel_size - 1].size() << endl;
    cout << "   input_tensor.size() = " << input_tensor.size() << endl;
    cout << "   output_tensor.size() = " << output_tensor.size() << endl;
    cout << "   output_tensor[" << output_tensor_channels - 1 << "][" << output_side_size - 1 << "].size() = " << output_tensor[output_tensor_channels - 1][output_side_size - 1].size() << endl;
    cout << "   ========================================" << endl;

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

double convolution::activation_function(double input_data)
{
    double output_data = 0.0;
    int this_node_dopped_out = 0;
    if (use_dopouts == 1)
    {
        double dropout_random = ((double)rand() / RAND_MAX);
        if (dropout_random < dropout_proportion)
        {
            this_node_dopped_out = 1;
        }
    }
    if (this_node_dopped_out == 0)
    {
        if (activation_function_mode == 0)
        {
            // 0 = sigmoid activation function
            output_data = 1.0 / (1.0 + exp(-input_data)); // Sigmoid function and put it into
        }
        else
        {
            // 1 = Relu simple activation function
            // 2 = Relu fix leaky activation function
            // 3 = Relu random variable leaky activation function
            if (input_data >= 0.0)
            {
                // Positiv value go right though ar Relu (Rectify Linear)
                output_data = input_data;
                // cout << "forward + output_data = " << output_data << endl;
            }
            else
            {
                // Negative
                switch (activation_function_mode)
                {
                case 1:
                    // 1 = Relu simple activation function
                    output_data = 0.0;
                    //  cout << "forward - output_data = " << output_data << endl;
                    break;
                case 2:
                    // 2 = Relu fix leaky activation function
                    output_data = input_data * fix_leaky_proportion;
                    //  cout << "forward -2 output_data = " << output_data << endl;
                    break;
                }
            }
        }
    }
    return output_data;
}
double convolution::delta_activation_func(double delta_outside_function, double value_from_node_outputs)
{
    double delta_inside_func = 0.0;
    if (activation_function_mode == 0)
    {
        // 0 = sigmoid activation function
        delta_inside_func = delta_outside_function * value_from_node_outputs * (1.0 - value_from_node_outputs); // Sigmoid function and put it into
    }
    else
    {
        // 1 = Relu simple activation function
        // 2 = Relu fix leaky activation function
        // 3 = Relu random variable leaky activation function
        if (value_from_node_outputs >= 0.0)
        {
            // Positiv value go right though ar Relu (Rectify Linear)
            delta_inside_func = delta_outside_function;
        }
        else
        {
            // Negative
            switch (activation_function_mode)
            {
            case 1:
                // 1 = Relu simple activation function
                delta_inside_func = 0;
                break;
            case 2:
                // 2 = Relu fix leaky activation function

                delta_inside_func = delta_outside_function * fix_leaky_proportion;
                break;
            }
        }
    }
    if (value_from_node_outputs == 0.0)
    {
        delta_inside_func = 0.0; // Dropout may have ocure
    }
    return delta_inside_func;
}

void convolution::conv_forward()
{
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        for (int y_slide = 0; y_slide < output_side_size; y_slide++)
        {
            for (int x_slide = 0; x_slide < output_side_size; x_slide++)
            {
                // Make the dot product of input tensor with kernel wheight for one kernel position
                double dot_product = kernel_bias_weights[out_ch_cnt]; // start with bias value for the dot product
                for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                {
                    for (int ky = 0; ky < kernel_size; ky++)
                    {
                        int inp_tens_y_pos = ky + y_slide * stride;
                        for (int kx = 0; kx < kernel_size; kx++)
                        {
                            int inp_tens_x_pos = kx + x_slide * stride;
                            // Itterate dot product
                            dot_product += input_tensor[in_ch_cnt][inp_tens_y_pos][inp_tens_x_pos] * kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx];
                        }
                    }
                }
                // Make the activation function
                // Put the dot product to output tensor map
                output_tensor[out_ch_cnt][y_slide][x_slide] = activation_function(dot_product);
            }
        }
    }
}
void convolution::xy_start_stop_transpose_conv(int slide_val)
{
    start_ret = kernel_size - slide_val - 1;
    if (start_ret < 0)
    {
        start_ret = 0;
    }
    int stop_ret = kernel_size + input_side_size - slide_val - 1;
    if (stop_ret > kernel_size)
    {
        stop_ret = kernel_size;
    }

    // Check and debug algorithm
    if (start_ret > kernel_size - 1)
    {
        cout << "debug algorithm start_ret = " << start_ret << endl;
        exit(0);
    }
    // Check and debug algorithm
    if (stop_ret < 1)
    {
        cout << "debug algorithm stop_ret = " << stop_ret << endl;
        exit(0);
    }
}

void convolution::conv_backprop()
{
    // Compute delta for each output channel
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        accum_bias_deltas[out_ch_cnt] = 0.0;
        for (int y_slide = 0; y_slide < output_side_size; y_slide++)
        {
            for (int x_slide = 0; x_slide < output_side_size; x_slide++)
            {
                // Compute derivative of activation function
                double delta_activation = delta_activation_func(o_tensor_delta[out_ch_cnt][y_slide][x_slide], output_tensor[out_ch_cnt][y_slide][x_slide]);
                internal_tensor_delta[out_ch_cnt][y_slide][x_slide] = delta_activation;
                // Update delta for bias weights
                accum_bias_deltas[out_ch_cnt] += delta_activation;
                // Update delta for kernel weights and input tensor
                for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                {
                    for (int ky = 0; ky < kernel_size; ky++)
                    {
                        int inp_tens_y_pos = ky + y_slide * stride;
                        for (int kx = 0; kx < kernel_size; kx++)
                        {
                            int inp_tens_x_pos = kx + x_slide * stride;
                            // Update delta for kernel weight
                            kernel_deltas[out_ch_cnt][in_ch_cnt][ky][kx] += delta_activation * input_tensor[in_ch_cnt][inp_tens_x_pos][inp_tens_y_pos];
                        }
                    }
                }
            }
        }
    }

    // Update delta for input tensor. Flipped 180 deg kernel_weight
    // TODO...
    //.....change below
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        for (int y_slide = 0; y_slide < input_side_size; y_slide++)
        {
            for (int x_slide = 0; x_slide < input_side_size; x_slide++)
            {
                for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                {
                    xy_start_stop_transpose_conv(y_slide);
                    for (int ky = start_ret; ky < stop_ret; ky = ky + stride)
                    {
                        xy_start_stop_transpose_conv(x_slide);
                        for (int kx = start_ret; kx < stop_ret; kx = kx + stride)
                        {
                            int inp_tens_y_pos = 0; // TODO
                            int inp_tens_x_pos = 0; // TODO.

                            // Update delta for input tensor. Flipped 180 deg kernel_weight
                            //                            i_tensor_delta[in_ch_cnt][inp_tens_x_pos][inp_tens_y_pos] += internal_tensor_delta[out_ch_cnt][y_slide][x_slide] * kernel_weights[out_ch_cnt][in_ch_cnt][kernel_size - ky - 1][kernel_size - kx - 1];
                    
                    //TODO....
                            i_tensor_delta[in_ch_cnt][inp_tens_x_pos][inp_tens_y_pos] += internal_tensor_delta[out_ch_cnt][y_slide][x_slide] * kernel_weights[out_ch_cnt][in_ch_cnt][kernel_size - ky - 1][kernel_size - kx - 1];
                        }
                    }
                }
            }
        }
    }
}
void convolution::conv_update_weights()
{
    // Update kernel weights
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        change_bias_weights[out_ch_cnt] = (learning_rate * accum_bias_deltas[out_ch_cnt]) + (momentum * change_bias_weights[out_ch_cnt]);
        kernel_bias_weights[out_ch_cnt] += change_bias_weights[out_ch_cnt];
        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
        {
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    change_weights[out_ch_cnt][in_ch_cnt][ky][kx] = (learning_rate * kernel_deltas[out_ch_cnt][in_ch_cnt][ky][kx]) + (momentum * change_weights[out_ch_cnt][in_ch_cnt][ky][kx]);
                    kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx] += change_weights[out_ch_cnt][in_ch_cnt][ky][kx];
                }
            }
        }
    }
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