# this is a function to create the dumbbell plot with two lists

import matplotlib.pyplot as plt

def dumbbell_plot(df, variable1, variable2, variable3):

    """
    df: dataframe, contain all the information, the index is the catoegories, or labels

    variable1: the first variable, the x-axis of the plot, the lower bound of the dumbbell
    variable2: the second variable, the x-axis of the plot, the upper bound of the dumbbell
    variable3: the third variable, the x-axis of the plot, the middle point of the dumbbell

    
    """
    # check if the variables are in the dataframe
    if variable1 not in df.columns:
        raise ValueError("variable1 is not in the dataframe")
    if variable2 not in df.columns:
        raise ValueError("variable2 is not in the dataframe")
    if variable3 not in df.columns:
        raise ValueError("variable3 is not in the dataframe")
    fig, ax = plt.subplots(figsize=(12,23))
    plt.hlines(y=range(df.shape[0]), xmin=df[variable1], xmax=df[variable2], color='grey', alpha=0.4, linewidth=5)
    plt.scatter(df[variable1], range(df.shape[0]), color='red', alpha=0.8, label=variable1, linewidths=20)
    plt.scatter(df[variable2], range(df.shape[0]), color='navy', alpha=0.8, label=variable2, linewidths=20)
    plt.scatter(df[variable3], range(df.shape[0]), color='green', alpha=0.8, label=variable3, linewidths=20)

    # Add title and axis names
    plt.yticks(range(df.shape[0]), df.index, size=40)
    plt.xticks(size=40)
    plt.xlim(0, 1.05)

    plt.show()