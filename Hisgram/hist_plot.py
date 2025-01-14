import matplotlib.pyplot as plt  
import pandas as pd
def plot_histogram(values, image_save_dir, variable_name):  
    # Create a histogram plot using the values  
    plt.hist(values, bins='auto', color='steelblue', edgecolor='black', alpha=0.7)  
  
    # Set the title, x-label, and y-label  
    plt.title(f'Histogram of {variable_name}')  
    plt.xlabel(variable_name)  
    plt.ylabel('Frequency')  
  
    # Save the histogram plot to the specified directory  
    plt.savefig(f'{image_save_dir}/{variable_name}_histogram.png')  
  
    # Show the histogram plot  
    plt.show()  
  
# Example usage:  
df = pd.read_excel('/Users/harrysun/Documents/Metadata/UMi/Copia de ROSE protocol data collection sheet completado.xlsx', sheet_name='Sheet1')
values = df['Age'].dropna().tolist()
image_save_dir = './'  
variable_name = 'Age'  
  
plot_histogram(values, image_save_dir, variable_name)