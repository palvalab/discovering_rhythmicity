import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib._cm import _Blues_data, _Reds_data

_Blues_data_fixed = list(_Blues_data)
_Blues_data_fixed[0] = [1.0, 1.0, 1.0]

_Reds_data_fixed = list(_Reds_data)
_Reds_data_fixed[0] = [1.0, 1.0, 1.0]

blues_fixed = LinearSegmentedColormap.from_list('Blues_fixed', _Blues_data_fixed, N=256)
blues_fixed_r = LinearSegmentedColormap.from_list('Blues_fixed_r', _Blues_data_fixed[::-1], N=256)

reds_fixed = LinearSegmentedColormap.from_list('Reds_fixed', _Reds_data_fixed, N=256)
reds_fixed_r = LinearSegmentedColormap.from_list('Reds_fixed_r', _Reds_data_fixed[::-1], N=256)


#### Colormap from our ripples paper
ripples_cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.166, 0.43, 0.43),
                   (0.33, 0.7, 0.7),
                   (0.5, 1.0, 1.0),
                   (0.66, 0.8, 0.8),
                   (1.0, 0.6, 0.6)),

         'green': ((0.0, 0.4, 0.4),
                   (0.166, 0.7, 0.7),
                   (0.33, 0.8, 0.8),
                   (0.5, 1.0, 1.0),
                   (0.66, 0.8, 0.8),
                   (1.0,0.0, 0.0)),

         'blue':  ((0.0, 0.8, 0.8),
                   (0.166, 1.0, 1.0),
                   (0.33, 1.0, 1.0),
                   (0.5, 0.4, 0.4),
                   (0.66, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

ripples_cmap = LinearSegmentedColormap('ripplescmap', ripples_cdict)

ripples_blue = ripples_cmap(0)
ripples_red = ripples_cmap(0.99)
ripples_orange = ripples_cmap(0.7)
ripples_orange2 = ripples_cmap(0.75)


# Okabe scheme
okabe = [ "#0072B2", "#D55E00","#CC79A7", "#009E73",  "#E69F00", "#56B4E9",    "#F0E442",  "#000000"]

ripples_blue = okabe[0]
noise_color = okabe[7]
ripples_red = okabe[2]
ripples_orange = okabe[1]
ripples_green = okabe[3]

### CB colors

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


### Subsystem colors, tab20c

subsystem_to_color = {'ContA': 0,
                      'ContB': 1,
                      'ContC': 2,
                              
                        'VisPeri': 4,
                         'VisCent': 5,
                                           
                          
                         'DorsAttnA': 8,
                         'DorsAttnB': 9,
                         'SalVentAttnA': 10,
                         'SalVentAttnB': 11,
                    
                         'DefaultA': 12,
                         'DefaultB': 13,
                         'DefaultC': 14,
                      
                         'SomMotA': 16,
                         'TempPar': 18,
                         'SomMotB': 17,
                         'Limbic': 19,
                         }


class CmapPruned(ListedColormap):
    def __init__(self, orig_cmap, prune_idx=[]):
        self.cmap = orig_cmap
        self.N = orig_cmap.N
        self.prune_idx = prune_idx

    def __call__(self, x, *args, **kwargs):
        res = self.cmap(x)

        res[self.prune_idx] = 1.0

        return res


def get_cmap_gamma_data(cmap_name, gamma=1.0):
    # this function is weird AF but works, maybe fix it later
    cmap_data = plt.get_cmap(cmap_name)(np.linspace(0, 1, 128))
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f'{cmap_name} {gamma}', cmap_data, 256, gamma=gamma)

    return new_cmap(np.linspace(0, 1, 128))