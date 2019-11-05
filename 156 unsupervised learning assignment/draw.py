import pandas as pd
import numpy as np
import seaborn as sns # statistical visualizations and aesthetics

from sklearn import linear_model
from sklearn import metrics
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import csv
py.sign_in("vickiichenn","9VEUuWa7Y5HHtpdum5Vo")

def scatter_with_color_dimension_graph(feature , target , layout_labels ) :
    '''Scatter with color dimension graph to visualize the density of the
    Given feature with target
    : param feature :
    : param target 
    : param layout labels :
    : return: '''

    trace1 = go.Scatter (
        y=feature,
        mode='markers',
        marker=dict(
                size=16,
                color=target,
                colorscale ='Viridis',
                showscale=True
        )
    )
    layout = go.Layout (
            title = layout_labels[2],
            xaxis=dict( title=layout_labels[0]) , yaxis=dict( title=layout_labels[ 1 ]))
    data = [ trace1 ]
    fig = Figure( data=data , layout=layout )
    py.image.save_as( fig , filename= layout_labels[ 1 ] + '_Density.png' )


def main () :
    glass_data = pd.read_csv('glass.csv', names=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'])
    glassType=['RI','Na','K','Mg','Al','Ca','Si','Ba','Fe']
    print(glassType[0])
    for i in glassType:
            graph_labels = [ " Number of Observations " , i , "Sample Glass Type Density Graph" ]
            scatter_with_color_dimension_graph( list ( glass_data[ i ] [ :215 ] ),
                                       np.array( [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7 ] ) ,
                                       graph_labels)
    df = pd.read_csv('glass_data_labeled.csv')
    features = ['Index','RI','Na','K','Mg','Al','Ca','Si','Ba','Fe','Type']
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
    xticklabels= features, yticklabels= features, alpha = 0.7,   cmap= 'coolwarm')
    plt.show()
if __name__ == '__main__':
    main()