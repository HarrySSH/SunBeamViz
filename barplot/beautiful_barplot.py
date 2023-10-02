import matplotlib.pyplot as plt  
import seaborn as sns  
  
def create_barplot(numbers, labels, sort_by='number', color_by='number', annotate=True, font_size=14, x_axis_font_size=14, y_axis_font_size=14, orientation='horizontal',
                   color_style="rocket"):  
    # Combine numbers and labels into a single list of tuples  

    color_style_list = ["viridis", "magma", "plasma","inferno",
                   "cividis", "rocket"]
    assert color_style in color_style_list, "the color theme is not in the color list provided"
    data = list(zip(labels, numbers))  
      
    # Sort the data based on the sort_by parameter  
    if sort_by == 'number':  
        data.sort(key=lambda x: x[1])  
    elif sort_by == 'label':  
        data.sort(key=lambda x: x[0])  
      
    # Unzip the sorted data  
    sorted_labels, sorted_numbers = zip(*data)  
      
    # Create a color palette  
    if color_by == 'number':  
        palette = sns.color_palette(color_style, len(numbers))  
        palette = [palette[i] for i in sorted(range(len(numbers)), key=lambda x: numbers[x])]  
    elif color_by == 'label':  
        palette = sns.color_palette(color_style, len(numbers))  
      
    # Create the barplot  
    sns.set(style="whitegrid")  
    plt.rcParams.update({'font.size': font_size})  
      
    if orientation == 'horizontal':  
        ax = sns.barplot(x=sorted_labels, y=sorted_numbers, palette=palette)  
    elif orientation == 'vertical':  
        ax = sns.barplot(x=sorted_numbers, y=sorted_labels, palette=palette)  
      
    # Set the font size for the x-axis and y-axis labels  
    ax.tick_params(axis='x', labelsize=x_axis_font_size)  
    ax.tick_params(axis='y', labelsize=y_axis_font_size)  
      
    # Annotate the barplot if annotate is True  
    if annotate:  
        for i, (label, number) in enumerate(zip(sorted_labels, sorted_numbers)):  
            if orientation == 'horizontal':  
                ax.text(i, number, f"{number}", ha='center', va='bottom')  
            elif orientation == 'vertical':  
                ax.text(number, i, f"{number}", ha='left', va='center')  
      
    # Show the barplot  
    plt.show()  
  
# Example usage  
numbers = [25, 30, 50, 15, 45]  
labels = ['A', 'B', 'C', 'D', 'E']  
  
create_barplot(numbers, labels, sort_by='number', color_by='number', annotate=True, font_size=14, x_axis_font_size=12, y_axis_font_size=12, orientation='vertical')  
