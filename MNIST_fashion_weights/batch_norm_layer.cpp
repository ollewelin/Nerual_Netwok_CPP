#include "batch_norm_layer.hpp"
#include <iostream>
#include <math.h>
#include <iostream>
#include <fstream>
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
    batch_size = 32; // 32 = default batch size
    channels = 0;
    rows = 0;
    cols = 0;
    setup_state = 0;
    init_gamma = 1.0;
    init_beta = 0.0;
}
void batch_norm_layer::get_version()
{
    cout << "batch_norm_layer version : " << version_major << "." << version_mid << "." << version_minor << endl;
    ver_major = version_major;
    ver_mid = version_mid;
    ver_minor = version_minor;
}

const int precision_to_text_file = 16;
void batch_norm_layer::load_weights(string filename)
{
    if (setup_state < 1)
    {
        cout << "Error could not load_weights() setup_state must be = 1 or more, setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        // TODO
        exit(0);
    }

    cout << "Load data weights ..." << endl;
    ifstream inputFile_w;
    inputFile_w.precision(precision_to_text_file);
    inputFile_w.open(filename);

    double d_element = 0.0;
    int data_load_error = 0;
    int data_load_numbers = 0;
    int data_b_load_numbers = 0;
    for (int gb = 0; gb < 2; gb++)
    {
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    if (inputFile_w >> d_element)
                    {

                        if (gb == 0) // 0= gamma load. 1= beta load
                        {
                            gamma[ch_idx][row_idx][col_idx] = d_element; // load data
                        }
                        else
                        {
                            beta[ch_idx][row_idx][col_idx] = d_element; // load data
                        }
                        data_load_numbers++;
                    }
                    else
                    {
                        data_load_error = 1;
                    }
                    if (data_load_error != 0)
                    {
                        break;
                    }
                }
                if (data_load_error != 0)
                {
                    break;
                }
            }
            if (data_load_error != 0)
            {
                break;
            }
        }
        if (data_load_error != 0)
        {
            break;
        }
    }
    inputFile_w.close();

    if (data_load_error == 0)
    {
        cout << "Load batch norm gamma beta weight data finnish !" << endl;
    }
    else
    {
        cout << "ERROR! weight file error have not sufficient amount of data to put into  all_weights[l_cnt][n_cnt][w_cnt] vector" << endl;
        cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
        cout << "Loaded this amout of data bias weights data_load_numbers = " << data_b_load_numbers << endl;
    }
    setup_state = 2;
}
void batch_norm_layer::save_weights(string filename)
{
    cout << "Save kernel weight data weights ..." << endl;
    ofstream outputFile_w;
    outputFile_w.precision(precision_to_text_file);
    outputFile_w.open(filename);
    for (int gb = 0; gb < 2; gb++)
    {
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {

                    if (gb == 0) // 0= gamma load. 1= beta load
                    {

                        outputFile_w << gamma[ch_idx][row_idx][col_idx] << endl;
                    }
                    else
                    {

                        outputFile_w << beta[ch_idx][row_idx][col_idx] << endl;
                    }
                }
            }
        }
    }
    outputFile_w.close();
    cout << "Save batch norm gamma beta weight data finnish !" << endl;
}
void batch_norm_layer::set_special_gamma_beta_init(double arg_gamma, double arg_beta)
{
    init_gamma = arg_gamma;
    init_beta = arg_beta;
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
    if (arg_channels > 0 && arg_channels < MAX_CHANNELS)
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
        delta_sum_gamma.push_back(dummy_2D_vector);
        delta_sum_beta.push_back(dummy_2D_vector);
        mean.push_back(dummy_2D_vector);
        variance.push_back(dummy_2D_vector);
    }
    for (int i = 0; i < batch_size; i++)
    {
        input_tensor.push_back(dummy_3D_vector);
        i_tensor_delta.push_back(dummy_3D_vector);
        output_tensor.push_back(dummy_3D_vector);
        o_tensor_delta.push_back(dummy_3D_vector);
        x_norm.push_back(dummy_3D_vector);
    }
    //============================================
    for (int ch_idx = 0; ch_idx < channels; ch_idx++)
    {
        for (int row_idx = 0; row_idx < rows; row_idx++)
        {
            for (int col_idx = 0; col_idx < cols; col_idx++)
            {
                delta_sum_gamma[ch_idx][row_idx][col_idx] = 0.0; // Clear for next batch update
                delta_sum_beta[ch_idx][row_idx][col_idx] = 0.0;  // Clear for next batch update
            }
        }
    }
    for (int ch_idx = 0; ch_idx < channels; ch_idx++)
    {
        for (int row_idx = 0; row_idx < rows; row_idx++)
        {
            for (int col_idx = 0; col_idx < cols; col_idx++)
            {
                gamma[ch_idx][row_idx][col_idx] = init_gamma; // Clear for next batch update
                beta[ch_idx][row_idx][col_idx] = init_beta;   // Clear for next batch update
            }
        }
    }
    setup_state = 1;
}

void batch_norm_layer::forward_batch(void)
{
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    if (sample_idx == 0)
                    {
                        // Clear, Zero the mean vector element now when this fucntion was called for the first time for this mini batch
                        mean[ch_idx][row_idx][col_idx] = 0.0;
                    }
                    // Calculate mean
                    double x = input_tensor[sample_idx][ch_idx][row_idx][col_idx];
                    mean[ch_idx][row_idx][col_idx] += (x - mean[ch_idx][row_idx][col_idx]);
                }
            }
        }
    }
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    if (sample_idx == 0)
                    {
                        mean[ch_idx][row_idx][col_idx] /= batch_size; // Complete mean calculation
                        variance[ch_idx][row_idx][col_idx] = 0.0;
                    }
                    // Calculate variance
                    double diff = input_tensor[sample_idx][ch_idx][row_idx][col_idx] - mean[ch_idx][row_idx][col_idx];
                    variance[ch_idx][row_idx][col_idx] += diff * diff;
                    if (sample_idx == batch_size - 1)
                    {
                        variance[ch_idx][row_idx][col_idx] /= batch_size; // complete variance calculation
                    }
                }
            }
        }
    }
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
                    // The x_norm_local variable represents the normalized input value.
                    double x = input_tensor[sample_idx][ch_idx][row_idx][col_idx];
                    double x_norm_local = (x - mean[ch_idx][row_idx][col_idx]) / (sqrt(variance[ch_idx][row_idx][col_idx]) + epsilon);
                    x_norm[sample_idx][ch_idx][row_idx][col_idx] = x_norm_local; // Stored for backpropagation
                    double y = gamma[ch_idx][row_idx][col_idx] * x_norm_local + beta[ch_idx][row_idx][col_idx];
                    output_tensor[sample_idx][ch_idx][row_idx][col_idx] = y;
                }
            }
        }
    }
}

void batch_norm_layer::backprop_batch(void)
{
    int data_size   = (batch_size*rows*cols);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    // Accumulate delta_sum_gamma and delta_sum_beta
                    delta_sum_gamma[ch_idx][row_idx][col_idx] += o_tensor_delta[sample_idx][ch_idx][row_idx][col_idx] * x_norm[sample_idx][ch_idx][row_idx][col_idx];
                    delta_sum_beta[ch_idx][row_idx][col_idx] += o_tensor_delta[sample_idx][ch_idx][row_idx][col_idx];
                }
            }
        }
    }
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
        for (int ch_idx = 0; ch_idx < channels; ch_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < cols; col_idx++)
                {
                    // Calcualte backprop o_tensor_delta, i_tesnor_delta
                    double xnorm    = x_norm[sample_idx][ch_idx][row_idx][col_idx];
                    double dgamma   = delta_sum_gamma[ch_idx][row_idx][col_idx];
                    double dbeta    = delta_sum_beta[ch_idx][row_idx][col_idx];
                    double gama_l   = gamma[ch_idx][row_idx][col_idx];
                    double o_delta  = o_tensor_delta[sample_idx][ch_idx][row_idx][col_idx];
                    double var      = variance[ch_idx][row_idx][col_idx];
                    i_tensor_delta[sample_idx][ch_idx][row_idx][col_idx] = gama_l * (o_delta - (dgamma * xnorm + dbeta) / data_size) / (sqrt(var) + epsilon);
                    // dx[i*D + j] = gamma[j] * (dout[i*D + j] - (dgamma_sum[j]*x_norm[i*D + j] + dbeta_sum[j])/(N*D)) / sqrt(var[j] + 1e-8);
                    if (sample_idx == batch_size - 1)
                    {
                        gamma[ch_idx][row_idx][col_idx] += lr * delta_sum_gamma[ch_idx][row_idx][col_idx];
                        beta[ch_idx][row_idx][col_idx] += lr * delta_sum_beta[ch_idx][row_idx][col_idx];
                    }
                }
            }
        }
    }
}

batch_norm_layer::~batch_norm_layer()
{
}
