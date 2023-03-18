
from utils.stats import get_module

def moving_average_fast(x, window_size):
    """
        Works in O(N) which is faster than convolution OR window view
    """
    xp = get_module(x)

    is_odd = (window_size % 2) == 1
    w_half = window_size // 2

    start = w_half + 1
    end = -w_half + 1

    tail_size = window_size
    tail_start = w_half - 1
    tail_end = -tail_size - 1

    if is_odd:
        tail_start += 1
        tail_end += 1
        end -= 1

    res = xp.zeros_like(x)

    x_cumsum = xp.cumsum(x, axis=-1)

    res[..., start:end] = (x_cumsum[..., tail_size:] - x_cumsum[..., :-tail_size])/window_size

    # fill head & tail, according to my tests it is faster than padding with zeros
    res[..., :w_half+1] = x_cumsum[..., tail_start:tail_size]/window_size
    res[..., -w_half:] = (x_cumsum[..., ~0:] - x_cumsum[..., tail_end:-w_half-1])/window_size
    
    return res
