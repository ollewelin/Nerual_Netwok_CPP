#include "batch_norm_layer.hpp"
#include <iostream>
using namespace std;

const int MAX_BATCH_SIZE = 256;
const int MAX_CHANNELS = 10000;
const int MAX_ROWS_OR_COLS = 100000;
batch_norm_layer::batch_norm_layer()
{
    cout << "Batch norm batch_norm_layer object constructor " << endl;
    version_major = 0;
    version_mid = 0;
    version_minor = 1;
    // 0.0.1 First empty version, untested.

    
    samp_cnt = 0;
    batch_size = 32;//32 = default batch size
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

void batch_norm_layer::set_up_tensors(int arg_batch_size, int arg_channels, int arg_rows, int arg_cols)//4D [batch_size][channels][row][col].
{
    if(arg_batch_size > 0 && arg_batch_size < MAX_BATCH_SIZE+1)
    {
        batch_size = arg_batch_size;
    }
    else
    {
        cout << "Error batch size argument out of range 1 to " << MAX_BATCH_SIZE << " arg_batch_size = " << arg_batch_size << endl;
        cout << "batch_size is default set to = " << batch_size << endl;
    }
    if(channels > 0 && channels < MAX_CHANNELS)
    {
        channels = arg_channels;
    }
    else
    {
        cout << "Error channels argument out of range 1 to " << MAX_CHANNELS  << " channels = " << channels << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    if(arg_rows > 0 && arg_rows < MAX_ROWS_OR_COLS)
    {
        rows = arg_rows;
    }
    else
    {
        cout << "Error rows argument out of range 1 to " << MAX_ROWS_OR_COLS  << " channels = " << arg_rows << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    if(arg_cols > 0 && arg_cols < MAX_ROWS_OR_COLS)
    {
        cols = arg_cols;
        if(rows != cols)
        {
            cout << "WARNINGS ! colums and rows are not set equals. Check this, cols = " << cols  << " rows = " << rows << endl;
        }
    }
    else
    {
        cout << "Error colums argument out of range 1 to " << MAX_ROWS_OR_COLS  << " channels = " << arg_cols << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    //========== Set up vectors below ============

    //============================================
}
int batch_norm_layer::forward_one_sample(void)
{
    if(samp_cnt < batch_size)
    {
        samp_cnt++;
    }
    else
    {
        samp_cnt = 0;
    }
    return samp_cnt;
}

batch_norm_layer::~batch_norm_layer()
{
}
