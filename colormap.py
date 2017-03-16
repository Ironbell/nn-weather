import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def YlOrRd(bins=5):
    """ 
        Yellow-Orange-Red colormap from colorbrewer2
    """
    colors_ = [(255,255,178), (254,204,92), (253,141,60), (240,59,32), (189,0,38)]  
    colors = [tuple(x / float(255) for x in t) for t in colors_]
    cmap_name = 'YlOrRd'
    return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=bins)
            
def RdYlGn(bins=5):
    """ 
        Red-Yellow-Green colormap from colorbrewer2
    """
    colors_ = [(215,25,28), (253,174,97), (255,255,191), (166,217,106), (26,150,65)]  
    colors = [tuple(x / float(255) for x in t) for t in colors_]
    cmap_name = 'YlOrRd'
    return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=bins)
            
def RdYlGn_r(bins=5):
    """ 
        Reversed Red-Yellow-Green colormap from colorbrewer2
    """
    colors_ = [(26,150,65), (166,217,106), (255,255,191), (253,174,97), (215,25,28)]  
    colors = [tuple(x / float(255) for x in t) for t in colors_]
    cmap_name = 'YlOrRd'
    return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=bins)
       