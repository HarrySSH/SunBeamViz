# Import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def piechart(number_list, cell_types, centre_circle=0.5, line_guide=False,
             image_name='piechart.png'):
    
    # Pie chart, where the slices will be ordered and plotted counter-clockwise
    # the label will be similar to the function above
    labels = cell_types
    sizes = number_list
    explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05,0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    # put a hole in the middle
    
    
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
    # add another not important color for the other cells
    
    
    explode = explode[:len(number_list)]
    COLORS = COLORS[:len(number_list)]
    
    if not line_guide:
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90, colors=COLORS)
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        
        plt.show()
    else:
        bbox_props=dict(boxstyle='square,pad=0.3',fc ='w',ec='k',lw=0.92)
        kw=dict(xycoords='data',textcoords='data',arrowprops=dict(arrowstyle='-'),zorder=0,va='center')

        fig1,ax1=plt.subplots(figsize=(20,8))
        
        values=number_list
        labels = [_celltype+'\n' + str(_value) + '' for _celltype, _value in zip(cell_types, values)]
        # Add code
        annotate_dict = zip(labels, values)
        annotate_dict_cp = zip(labels, values)
        
        # color as well
        COLORS_dict =  zip(COLORS, values)
        COLORS_dict_cp =  zip(COLORS, values)
        
        val = [[x,y] for x,y in zip(sorted(values, reverse=True),sorted(values))]
        annotated_list = []
        for x,y in zip(sorted(annotate_dict, key=lambda x: x[1], reverse=True),  
                                               sorted(annotate_dict_cp, key=lambda x: x[1])):
            
            annotated_list.append([x,y])
            

        color_list = []
        for x,y in zip(sorted(COLORS_dict, key=lambda x: x[1], reverse=True),  
                                               sorted(COLORS_dict_cp, key=lambda x: x[1])):
            
            color_list.append([x,y])
            
        
            
        assert len(annotated_list) !=0, 'annotated_list is empty'


        # conver the color as well
        
        values1 = sum(val, [])
        annotate_dict = sum(annotated_list, [])
        color_dict = sum(color_list, [])
        
        new_labels = []
        new_colors = [] 
        
        for i in range(len(values)):
            
            new_labels.append(annotate_dict[i][0])
            #print(color_list[i])
            #print(color_list[i][0])
            new_colors.append(color_dict[i][0])
        #sdvs
        
                    
        wedges,texts=ax1.pie(values1[:len(values)],labeldistance=1.2,startangle=90,
                             colors=new_colors,
                             
                               explode=explode)
        
        for i,p in enumerate(wedges):
            if 'Other' in new_labels[i]:
                continue
            ang=(p.theta2-p.theta1)/2. +p.theta1
            y=np.sin(np.deg2rad(ang))
            x=np.cos(np.deg2rad(ang))
             
            horizontalalignment={-1:"right",1:"left"}[int(np.sign(x))]
            
            connectionstyle="angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle":connectionstyle})
            ax1.annotate(new_labels[i],xy=(x, y),xytext=(1.35*np.sign(x),1.4*y),
                        horizontalalignment=horizontalalignment,**kw, fontsize=30,
                        weight='bold')
        
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # add how many sample are there as text 
        ax1.text(0, 0, str(sum(sizes)), ha='center', va='center', fontsize=30, weight='bold')
            
        
        plt.savefig(image_name, dpi=300)

'''
Adenocarcinoma
170
Benign
76
Non-small cell carcinoma
23
Squamous cell carcinoma
76
Large cell carcinoma
1
'''


number_list = [170, 76, 23, 76, 1]
category_list = ['Adenocarcinoma', 'Benign', 'Non-small cell carcinoma', 'Squamous cell carcinoma', 'Large cell carcinoma']



piechart(number_list, category_list, centre_circle=0.8, line_guide=True, image_name='Tumor_profiling.png')


