import matplotlib.pyplot as plt

class FrequencyCommunity:
    @staticmethod
    def from_blob(blob, f_vals):
        freq_mid = f_vals[int(blob[0])]
        radius = blob[2]
        
        return FrequencyCommunity(freq_mid, radius)
    
    def __init__(self, center, radius, scale=1.05):
        self.center = center
        self.freq_lower = center*(1/scale**radius)
        self.freq_upper = center*(scale**radius)
        self.size = self.freq_upper - self.freq_lower
        
    def plot(self, ax, color='red', lw=2, ls='--'):
        rect_black = plt.Rectangle([self.freq_lower, self.freq_lower], self.size, self.size, facecolor='none', edgecolor=color, lw=lw, ls=ls, zorder=11)
        rect_white = plt.Rectangle([self.freq_lower, self.freq_lower], self.size, self.size, facecolor='none', edgecolor='white', lw=lw+0.5, zorder=10)
        ax.add_patch(rect_black)
        ax.add_patch(rect_white)