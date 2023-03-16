#include "batch_norm_layer.hpp"

batch_norm_layer::batch_norm_layer()
{
    samp_cnt = 0;
}
int batch_norm_layer::set_in_tensor(vector<vector<vector<double>>> input_tensor)
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
void batch_norm_layer::forward_one_sample(void)
{
}

batch_norm_layer::~batch_norm_layer()
{
}
