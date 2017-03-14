import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def YlOrRd():
    """ 
        Yellow or Red colormap from colorbrewer2
    """
    colors_ = [(255,255,178), (254,204,92), (253,141,60), (240,59,32), (189,0,38)]  
    colors = [tuple(x / float(255) for x in t) for t in colors_]
    cmap_name = 'YlOrRd'
    return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=5)
       