#include "batch_norm_layer.hpp"
#include <iostream>
#include <math.h>
using namespace std;

const int MAX_BATCH_SIZE = 256;
const int MAX_CHANNELS = 10000;
const int MAX_ROWS_OR_COLS = 100000;
const double epsilon = 1e-8;
batch_norm_layer::batch_norm_layer()
{
    cout << "Batch norm batch_norm_layer object constructor " << endl;
    version_major = 0;
    version_mid = 0;
    version_minor = 1;
    // 0.0.1 First empty version, untested.

    samp_cnt = 0;
    batch_size = 32; // 32 = default batch size
    channels = 0;
    rows = 0;
    cols = 0;
}
void batch_norm_layer::get_version()
{
    cout << "batch_norm_layer version : " << version_major << "." << version_mid << "." << version_minor << endl;
    ver_major = version_major;
    ver_mid = version_mid;
    ver_minor = version_minor;
}

void batch_norm_layer::set_up_tensors(int arg_batch_size, int arg_channels, int arg_rows, int arg_cols) // 4D [batch_size][channels][row][col].
{
    if (arg_batch_size > 0 && arg_batch_size < MAX_BATCH_SIZE + 1)
    {
        batch_size = arg_batch_size;
    }
    else
    {
        cout << "Error batch size argument out of range 1 to " << MAX_BATCH_SIZE << " arg_batch_size = " << arg_batch_size << endl;
        cout << "batch_size is default set to = " << batch_size << endl;
    }
    if (channels > 0 && channels < MAX_CHANNELS)
    {
        channels = arg_channels;
    }
    else
    {
        cout << "Error channels argument out of range 1 to " << MAX_CHANNELS << " channels = " << channels << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    if (arg_rows > 0 && arg_rows < MAX_ROWS_OR_COLS)
    {
        rows = arg_rows;
    }
    else
    {
        cout << "Error rows argument out of range 1 to " << MAX_ROWS_OR_COLS << " channels = " << arg_rows << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    if (arg_cols > 0 && arg_cols < MAX_ROWS_OR_COLS)
    {
        cols = arg_cols;
        if (rows != cols)
        {
            cout << "WARNINGS ! colums and rows are not set equals. Check this, cols = " << cols << " rows = " << rows << endl;
        }
    }
    else
    {
        cout << "Error colums argument out of range 1 to " << MAX_ROWS_OR_COLS << " channels = " << arg_cols << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    //========== Set up vectors below ============
    vector<double> dummy_1D_vector;
    vector<vector<double>> dummy_2D_vector;
    vector<vector<vector<double>>> dummy_3D_vector;
    for (int i = 0; i < cols; i++)
    {
        dummy_1D_vector.push_back(0.0);
    }
    for (int i = 0; i < rows; i++)
    {
        dummy_2D_vector.push_back(dummy_1D_vector);
    }
    for (int i = 0; i < channels; i++)
    {
        dummy_3D_vector.push_back(dummy_2D_vector);
        gamma.push_back(dummy_2D_vector);
        beta.push_back(dummy_2D_vector);
        delta_gamma.push_back(dummy_2D_vector);
        delta_beta.push_back(dummy_2D_vector);
        mean.push_back(dummy_2D_vector);
        variance.push_back(dummy_2D_vector);
    }
    for (int i = 0; i < batch_size; i++)
    {
        input_tensor.push_back(dummy_3D_vector);
        i_tensor_delta.push_back(dummy_3D_vector);
        output_tensor.push_back(dummy_3D_vector);
        o_tensor_delta.push_back(dummy_3D_vector);
    }
    //============================================
}

// 1. Call this function ever sample thoruge a batch to calcualte mean and varaince
// 2. When this fucntion have been called though all samples of a mini batch it will automaticly
//    do the batch norm forward and then calculated the output_tensor vector with respect to the calculated mean, varaince and the gamma, beta learned parameters
int batch_norm_layer::forward_batch(void)
{
    samp_cnt++;
    if (samp_cnt < batch_size)
    {
        // 1. Call this function ever sample thoruge a batch to calcualte mean and varaince
        // Do the mean and varaince calcualtion here for one sample at a time
        // Calcualte and store the mean and varaince in the vectors mean and variance how already are set up outside this fucntion
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    if (samp_cnt == 1)
                    {
                        // Clear, Zero the mean vector element now when this fucntion was called for the first time for this mini batch
                        mean[ch_idx][row_idx][col_idx] = 0.0;
                    }
                    // Calculate mean
                    double x = input_tensor[samp_cnt][ch_idx][row_idx][col_idx];
                    mean[ch_idx][row_idx][col_idx] += (x - mean[ch_idx][row_idx][col_idx]);
                }
            }
        }
    }
    else
    {
        samp_cnt = 0; // Tell the caller of this fucntion that the batch norm calculation is finnish when samp_cnt = 0
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
        {
            for (int ch_idx = 0; ch_idx < channels; ch_idx++)
            {
                for (int row_idx = 0; row_idx < rows; row_idx++)
                {
                    for (int col_idx = 0; col_idx < cols; col_idx++)
                    {
                        mean[ch_idx][row_idx][col_idx] /= batch_size;
                        // Calculate variance
                        double diff = input_tensor[sample_idx][ch_idx][row_idx][col_idx] - mean[ch_idx][row_idx][col_idx];
                        variance[ch_idx][row_idx][col_idx] = diff * diff;
                    }
                }
            }
        }
        // 2. When this fucntion have been called though all samples of a mini batch it will automaticly
        // do the batch norm forward and then calculated the output_tensor vector with respect to the
        // Use the stored inforrmation data in this vectors
        // gamma[ch_idx][row_idx][col_idx]
        // beta[ch_idx][row_idx][col_idx]
        // mean[ch_idx][row_idx][col_idx]
        // variance[ch_idx][row_idx][col_idx]
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
        {
            for (int ch_idx = 0; ch_idx < channels; ch_idx++)
            {
                for (int row_idx = 0; row_idx < rows; row_idx++)
                {
                    for (int col_idx = 0; col_idx < cols; col_idx++)
                    {
                        // Here, epsilon is a small constant added to the variance to avoid division by zero.
                        // The x_hat variable represents the normalized input value.
                        double x = input_tensor[sample_idx][ch_idx][row_idx][col_idx];
                        double x_hat = (x - mean[ch_idx][row_idx][col_idx]) / sqrt(variance[ch_idx][row_idx][col_idx] / (samp_cnt - 1) + epsilon);
                        double y = gamma[ch_idx][row_idx][col_idx] * x_hat + beta[ch_idx][row_idx][col_idx];
                        output_tensor[sample_idx][ch_idx][row_idx][col_idx] = y;
                    }
                }
            }
        }
    }
    return samp_cnt;
}

int batch_norm_layer::backprop_batch(void)
{
    samp_cnt++;
    if (samp_cnt < batch_size)
    {
        // 1. Call this function ever sample thoruge a batch to calcualte to calculate every sample delta backpropagataion though this batch norm layer
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    // TODO..
                    // Calcualte backprop o_tensor_delta, i_tesnor_delta
                    //  i_tensor_delta[sample_idx][ch_idx][row_idx][col_idx] =  <insert code here>  o_tensor_delta[sample_idx][ch_idx][row_idx][col_idx];
                    
                    //TODO... accumulate delta_gamma and delta_beta 

                }
            }
        }
    }
    else
    {
        samp_cnt = 0; // Tell the caller of this fucntion that the batch norm calculation is finnish when samp_cnt = 0

        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
        {
            for (int ch_idx = 0; ch_idx < channels; ch_idx++)
            {
                for (int row_idx = 0; row_idx < rows; row_idx++)
                {
                    for (int col_idx = 0; col_idx < cols; col_idx++)
                    {
                        // TODO..
                        //gamma
                    }
                }
            }
        }
    }
    return samp_cnt;
}

batch_norm_layer::~batch_norm_layer()
{
}
