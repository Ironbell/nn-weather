import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
       