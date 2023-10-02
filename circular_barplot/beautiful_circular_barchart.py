# Import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_circular_barchart(number_list, cell_types, labelPadding=1):
    # initialize the figure
    plt.figure(figsize=(20,10))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')


    lowerLimit = 5
    # Compute max and min in the dataset
    _max = max(number_list)
    _min = min(number_list)

    import math

    slope = (_max - _min) / _max
    heights = [math.log2(slope * x +3)  for x in number_list]

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(cell_types)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(cell_types)+1))
    angles = [element * width for element in indexes] 
    COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#1f1f1f",
            "#ff0000",
            '#d3d3d3',
        ]
    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        # color is as it is if the number is positive, otherwise it is gray
        color=[COLORS[i] if number_list[i] > 0 else "#d3d3d3" for i in range(len(cell_types))],
        edgecolor="white")


    # Add labels
    for bar, angle, height, label, number in zip(bars,angles, heights, cell_types, number_list):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180 
        else: 
            alignment = "left" 
            rotation = rotation 

        # Finally add the labels, the angle should be vertical  to the bar:
        rotation = rotation + 90
        ax.text(

            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding , 
            s=label, #+'\n'+str(number), 
            ha=alignment, 
            va='top', 
            rotation=rotation, 
            ### make the label at the center of the bar

            rotation_mode="anchor",
            weight="bold",
        size = 20) 
        # number is bold 
        ax.text(
            x=angle, 
            y=lowerLimit + min(heights), 
            s=str(number), 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor",
            weight="bold",
        size = 20) 
    plt.tight_layout()
    #plt.show()
        
    plt.savefig('beautiful_circular_barchart.png')


    

        


cell_types =['Myeloblast', 'Promyelocyte', 'Myelocyte','Metamyelocyte',
                      "Band\nneutrophil", "Segmented\nneutrophil",
                      'Eosinophil', 
                     'Basophil', 'Monocyte', 'Lymphocyte', 'Plasma cell', 'Normoblast', 'Others']

labelPadding = 1
number_list = [9, 0,0,1,2,1, 1, 0, 1, 4, 0, 2,1]
plot_circular_barchart(number_list, cell_types)
# 12 number sum to 100
number_list = [47,2,3,2, 7, 7, 3, 1, 4, 15, 2,6, 1]






