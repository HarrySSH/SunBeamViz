# this is a function to create the dumbbell plot with two lists

import matplotlib.pyplot as plt

def dumbbell_plot(df, variable1, variable2):
    fig, ax = plt.subplots(figsize=(12,23))
    plt.hlines(y=range(df.shape[0]), xmin=df[variable1], xmax=df[variable2], color='grey', alpha=0.4, linewidth=5)
    plt.scatter(df[variable1], range(df.shape[0]), color='red', alpha=0.8, label=variable1, linewidths=20)
    plt.scatter(df[variable2], range(df.shape[0]), color='navy', alpha=0.8, label=variable2, linewidths=20)

    # Add title and axis names
    plt.yticks(range(df.shape[0]), df.index, size=40)
    plt.xticks(size=40)
    plt.xlim(0, 1.05)

    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    CS = [0.905982905982906, 0.8414557228155747, 0.9601920768307323, 0.9489585834333734, 0.814513305322129, 0.7543178720504301, 0.8930234859480142, 0.9740385515908491, 0.9489710884353743, 0.9603846153846154, 0.9162745098039216, 0.9391666666666666, 0.7189278227014075, 0.8398343036274071, 0.8327505827505828, 0.8528815004262575, 0.8377356909543369, 0.9032312925170067, 0.917264799536836, 0.9901960784313726, 0.9844812925170068, 0.9134437519565155, 0.9257490230134607]
    exp = [0.8413075296796227, 0.7867686527106817, 0.8514333542582131, 0.9335580386000553, 0.651533981163999, 0.6222843822843823, 0.5607042817126852, 0.7556878306878306, 0.8820634920634921, 0.8995358090185676, 0.8258634006399647, 0.873805256869773, 0.5543209876543209, 0.5275807722616233, 0.5682957393483709, 0.49969310509445536, 0.7118480725623583, 0.8182716049382716, 0.7922034061851692, 0.9242424242424242, 0.9722222222222223, 0.7834343434343434, 0.8632756132756132]
    cell_types = ['B1', 'B2', 'E1', 'E4', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6', 'L2', 'L4', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'MO2', 'PL2', 'PL3', 'U1', 'U4']
    data = {'AI': CS,
        'human': exp,
        'cell_types': cell_types}

    df = pd.DataFrame(data, index=cell_types)

    dumbbell_plot(df, 'AI', 'human')


