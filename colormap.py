import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def colormap(colors, bins, cmap_name):
    colors_ = [tuple(x / float(255) for x in t) for t in colors]
    return LinearSegmentedColormap.from_list(
            cmap_name, colors_, N=bins)

def YlOrRd(bins=5):
    """ 
        Yellow-Orange-Red colormap from colorbrewer2
    """
    colors = [(255,255,178), (254,204,92), (253,141,60), (240,59,32), (189,0,38)] 
    return colormap(colors, bins, 'YlOrRd')
     
def RdYlGn(bins=5):
    """ 
        Red-Yellow-Green colormap from colorbrewer2
    """
    colors = [(215,25,28), (253,174,97), (255,255,191), (166,217,106), (26,150,65)]  
    return colormap(colors, bins, 'RdYlGn')
     
def RdYlGn_r(bins=5):
    """ 
        Reversed Red-Yellow-Green colormap from colorbrewer2
    """
    colors = [(26,150,65), (166,217,106), (255,255,191), (253,174,97), (215,25,28)]  
    return colormap(colors, bins, 'RdYlGn_r')
            
def RdYlBu(bins=6):
    """ 
        Red-Yellow-Blue colormap from colorbrewer2 (colorblind safe!)
    """
    colors = [(215,25,28), (253,174,97), (254,224,144), (255,255,191), (171,217,233), (44,123,182)] 
    return colormap(colors, bins, 'RdYlBu')    

def RdYlBu_r(bins=6):
    """ 
        Reverse Red-Yellow-Blue colormap from colorbrewer2 (colorblind safe!)
    """
    colors = [(44,123,182),(171,217,233), (255,255,191), (254,224,144), (253,174,97), (215,25,28)] 
    return colormap(colors, bins, 'RdYlBu_r')    
    
def BuRd(bins=7):
    """ 
        2 concatenated single-hue colormaps from colorbrewer2
    """
    colors = [(33, 113, 181), (107, 174, 214), (189, 215, 231), (255,255,255), (252, 174, 145), (251, 106, 74), (203, 24, 29)] 
    return colormap(colors, bins, 'BuRd')   
    
def BuRd_r(bins=7):
    """ 
        (reversed) 2 concatenated single-hue colormaps from colorbrewer2
    """
    colors = [(203, 24, 29),(251, 106, 74),(252, 174, 145),(255,255,255),(189, 215, 231),(107, 174, 214),(33, 113, 181) ] 
    return colormap(colors, bins, 'BuRd_r')   
    
def Gn(bins=5):
    """ 
        single-hue colormap (green) from colorbrewer2
    """
    colors = [(237, 248, 233),(186, 228, 179),(116, 196, 118),(49, 163, 84),(0, 109, 44)] 
    return colormap(colors, bins, 'Gn')  
    
def Gn_r(bins=5):
    """ 
        (reversed) single-hue colormap (green) from colorbrewer2
    """
    colors = [(0, 109, 44),(49, 163, 84),(116, 196, 118),(186, 228, 179),(237, 248, 233)] 
    return colormap(colors, bins, 'Gn_r')  
    
def Dark(bins=3):
    """
        qualitative colormap from colorbrewer2
    """
    colors = [(27,158,119), (217,95,2), (117,112,179)]
    return colormap(colors, bins, 'Dark')  
       