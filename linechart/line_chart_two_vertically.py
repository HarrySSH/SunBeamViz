import matplotlib.pyplot as plt
import numpy as np

# Sample data for the two lists (up and down)
# the goal is to create two line chart with the same x axis
# and the same or different y axis, but with different colors

# the x axis is the same
def line_chart_two_variables(res_1, res_2, x_axis, label_1, label_2, dash_interval):
    y_up = res_1
    y_down = res_2

    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(8, 6))

    # Customize the appearance of the up line chart
    ax_up.plot(x_axis, y_up,marker ='o', color='brown', linestyle='--', linewidth=4)
    ax_up.set_ylabel(label_1,
        size =30, fontweight='bold')
    
    # since we put the plot vertically, and they share the same axis,
    # we don't need to put the x axis for the upper plot
    ax_up.set_xticks([])

    for tick in ax_up.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
    for label in ax_up.get_yticklabels():
            label.set_fontweight('bold')
    ax_up.legend()

    # Customize the appearance of the down line chart
    ax_down.plot(x_axis, y_down, 
             marker ='o',color='pink', linestyle='--', linewidth=4)
    ax_down.set_xlabel('Iteration',size =30, fontweight='bold')
    ax_down.set_ylabel('Eval Loss', size =30, fontweight='bold')
    ax_down.legend()

    for tick in ax_down.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
        
    for tick in ax_down.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for label in ax_down.get_yticklabels():
            label.set_fontweight('bold')
    for label in ax_down.get_xticklabels():
            label.set_fontweight('bold')

    # add the dash line
    dash_interval = [x for x in range(0, len(x_axis), dash_interval)]
    for _ in dash_interval:
        ax_up.axvline(_, color='grey', linestyle='--', linewidth=1)
        ax_down.axvline(_, color='grey', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


# main function


import math
learning_rate = [(0.001 * (math.cos(math.pi * (x%10) / 10) + 1) / 2) for x in range(50)]
eval_loss = [0.08681869, 0.06113231,0.042925708, 0.04829566, 0.0420778, 0.037360527, 0.032510455,
 0.040861417, 0.04225681,0.041545182,0.06695634,0.062669374, 0.04887452,0.05779295,
 0.05072676,0.043129697,0.048170723,0.054728527,0.03798419,0.034259054,0.06549044,0.051102335,0.04739002,0.04083165,0.034363454,
 0.03606452, 0.042497363,0.035552867,0.039134266,0.03156854,0.06127965,0.04568231,0.04623363,0.039629553,0.0272555,0.042644117,
 0.03964063,0.041390903, 0.03054502,0.030448443,0.0529789,0.045372646, 0.03896754, 0.04190331,0.039587617,0.040453467,0.037071215,
 0.03440639, 0.033938447, 0.031225332,
]

x_axis = [x for x in range(50)]
label_1 = 'Learning Rate'
label_2 = 'Eval Loss'
dash_interval = 10

line_chart_two_variables(learning_rate, eval_loss, x_axis, label_1, label_2, dash_interval)


    

    








