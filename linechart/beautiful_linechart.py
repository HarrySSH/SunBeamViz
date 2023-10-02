import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
  
def create_lineplot(data, labels, sort_by='number', color_by='number', font_size=14, x_axis_font_size=14, y_axis_font_size=14, color_style="rocket"):  
    color_style_list = ["viridis", "magma", "plasma", "inferno", "cividis", "rocket"]  
    assert color_style in color_style_list, "the color theme is not in the color list provided"  
      
    # Create a DataFrame from the data and labels  
    df = pd.DataFrame(data, columns=labels)  
      
    # Sort the DataFrame based on the sort_by parameter  
    if sort_by == 'number':  
        df = df.sort_values(by=labels, ascending=True)  
    elif sort_by == 'label':  
        df = df[sorted(labels, key=lambda x: x)]  
      
    # Create a color palette  
    if color_by == 'number':  
        palette = sns.color_palette(color_style, len(labels))  
    elif color_by == 'label':  
        palette = {label: sns.color_palette(color_style, len(labels))[i] for i, label in enumerate(labels)}  
      
    # Create the lineplot  
    sns.set(style="whitegrid")  
    plt.rcParams.update({'font.size': font_size})  
      
    ax = sns.lineplot(data=df, palette=palette, dashes=False)  
      
    # Set the font size for the x-axis and y-axis labels  
    ax.tick_params(axis='x', labelsize=x_axis_font_size)  
    ax.tick_params(axis='y', labelsize=y_axis_font_size)  
      
    # Show the lineplot  
    plt.show()  
  
# Example usage  
data = [  
    [25, 30, 50, 15, 45],  
    [28, 35, 48, 20, 40],  
    [22, 38, 55, 12, 50]  
]  
labels = ['A', 'B', 'C', 'D', 'E']  
  
create_lineplot(data, labels, sort_by='number', color_by='number', font_size=14, x_axis_font_size=12, y_axis_font_size=12)  
