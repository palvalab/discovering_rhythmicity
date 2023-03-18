import cupy as cp

_pac_kernel = cp.ElementwiseKernel(
             'T x, raw T y, raw I lags, int32 n_cols', 'raw C output',
             '''
             int curr_col = i % n_cols;
             int curr_row = i / n_cols;
             
             int sample_lag = lags[curr_row];
             int offset_idx = curr_col + sample_lag;
             
             if(offset_idx < n_cols) {
                 int compare_idx = curr_row*n_cols + offset_idx;
                 output[i] = x * y[compare_idx];
             }
             else
             {
                 output[i] = __int_as_float(0xFFE00000);
             }
                                       
             ''',
             '_pac_kernel',
            )