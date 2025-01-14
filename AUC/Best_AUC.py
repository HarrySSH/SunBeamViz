import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
import argparse
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from scipy.stats import rankdata
# Example data
np.random.seed(42)
def create_auc_plot(y_true, y_scores, save_path):
    # Compute ROC curve and AUC
    ranks = rankdata(y_scores)
    y_scores = ranks / len(ranks)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # remove the first threshold, which is inf 
    ## for all the other thresdhold set more interval, 10 itnerval for each threshold

    thresholds = np.concatenate([np.linspace(thresholds[i], thresholds[i+1], 10, endpoint=False) for i in range(1,len(thresholds)-1)])
    # add the last threshold
    thresholds = np.concatenate([thresholds, [0]])
    fpr = []
    tpr = []
    for threshold in tqdm(thresholds, desc='Calculating all the possible fpr and tpr'):
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
    #raise NotImplementedError("stop")   
    



    # Create the dot plot
    def create_dot_plot(threshold, y_true, y_scores):
        df = pd.DataFrame({'y_true': y_true, 'y_scores': y_scores})
        df['predicted'] = (df['y_scores'] >= threshold).astype(int)
        
        actual_positive = df[df['y_true'] == 1]
        actual_negative = df[df['y_true'] == 0]
        
        true_positive = actual_positive[actual_positive['predicted'] == 1]
        false_negative = actual_positive[actual_positive['predicted'] == 0]
        true_negative = actual_negative[actual_negative['predicted'] == 0]
        false_positive = actual_negative[actual_negative['predicted'] == 1]
        
        return [
            go.Scatter(x=true_positive['y_scores'], y=np.ones(len(true_positive)), mode='markers', name=f'TP: {len(true_positive)}',
                       marker_color='blue'),
            go.Scatter(x=false_negative['y_scores'], y=np.ones(len(false_negative)), mode='markers', name=f'FN: {len(false_negative)}',
                        marker_color='lightblue'),
            go.Scatter(x=true_negative['y_scores'], y=np.zeros(len(true_negative)), mode='markers', name=f'TN: {len(true_negative)}',
                       marker_color='red'),
            go.Scatter(x=false_positive['y_scores'], y=np.zeros(len(false_positive)), mode='markers', name=f'FP: {len(false_positive)}',
                       marker_color='lightcoral'),
            # add a text to show the sensitivity and specificity
            go.Scatter(x=[threshold], y=[-0.1], mode='text', text=[f'Sensitivity: {len(true_positive) / (len(true_positive) + len(false_negative)):.3f}'], showlegend=False),
            go.Scatter(x=[threshold], y=[-0.2], mode='text', text=[f'Specificity: {len(true_negative) / (len(true_negative) + len(false_positive)):.3f}'], showlegend=False),
  
            go.Scatter(x=[threshold, threshold], y=[-0.1, 1.1], mode='lines', name='Threshold', line=dict(color='black', dash='dash'))
        ]

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC Curve', 'Dot Plot'))

    # Add ROC curve and moving dot as a single trace
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers', name=f'ROC curve (AUC = {roc_auc:.3f})', 
                            line=dict(color='darkorange'), 
                            marker=dict(size=[0]*len(fpr), color=['darkorange']*len(fpr), 
                                        symbol=['circle']*len(fpr))), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')), row=1, col=1)

    # Add initial dot plot to the subplot
    initial_dot_plot_traces = create_dot_plot(thresholds[0], y_true, y_scores)
    for trace in initial_dot_plot_traces:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        xaxis_title="Specificity",
        yaxis_title="Sensitivity",
        xaxis2_title="Predicted Score",
        yaxis2=dict(title="Class", tickvals=[0, 1], ticktext=['Negative', 'Positive']),
    )

    # Create frames for animation
    frames = []
    for i, threshold in enumerate(thresholds):
        marker_sizes = [0] * len(fpr)
        marker_sizes[i] = 15
        marker_colors = ['darkorange'] * len(fpr)
        marker_colors[i] = 'green'
        
        frame = go.Frame(
            data=[
                go.Scatter(x=fpr, y=tpr, marker=dict(size=marker_sizes, color=marker_colors)),  # Update ROC curve and dot
                go.Scatter(visible=True),  # Random line (unchanged)
            ] + create_dot_plot(threshold, y_true, y_scores),  # Update dot plot
            name=f'{threshold:.2f}'
        )
        
        
        frame['layout'] = go.Layout(
            annotations=[
                dict(
                    x=0.25, y=1.05,
                    xref='paper', yref='paper',
                    text=f"Threshold: {threshold:.3f}, FPR: {fpr[i]:.3f}, TPR: {tpr[i]:.3f}",
                    showarrow=False
                )
            ]
        )
        frames.append(frame)

    fig.frames = frames


    # Add slider
    sliders = [dict(
        active=0,
        yanchor='top',
        xanchor='left',
        currentvalue=dict(
            font=dict(size=16),
            prefix='Threshold: ',
            visible=True,
            xanchor='right'
        ),
        transition={'duration': 0},
        pad={'b': 10, 't': 50},
        len=0.9,
        x=0.1,
        y=0,
        steps=[dict(
            args=[
                [f'{threshold:.2f}'],
                dict(frame=dict(duration=0, redraw=True), mode='immediate')
            ],
            label=f'{threshold:.2f}',
            method='animate'
        ) for threshold in thresholds]
    )]

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=0, redraw=True), 
                                        fromcurrent=True, 
                                        mode='immediate',
                                        transition=dict(duration=0))]),
                    dict(label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                            mode='immediate',
                                            transition=dict(duration=0))])]
        )],
        sliders=sliders
    )

    # Display the interactive plot
    fig.show()
    print("AUC: ", roc_auc)
    print("dir", save_path)
    # Save the plot for viewing
    if save_path:
        fig.write_html(save_path)
    else:
        fig.write_html("roc_curve_and_dot_plot.html")

def main():
    parser = argparse.ArgumentParser(description='Create an interactive ROC curve and dot plot.')
    
    # stimutlate a set of y_true and y_scores, the AUC is around 0.8
    parser.add_argument('--input', type=str, help='Path to the input file containing y_true and y_scores')

    parser.add_argument('--save_path', type=str, help='Path to save the plot as an HTML file')

    args = parser.parse_args()

    # if end with csv, read_csv, else read_pickle
    try:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        elif args.input.endswith('.pkl'):
            df = pd.read_pickle(args.input)
        else:
            NotImplementedError("Only support csv and pkl file")
    except:
        # check the foramt, the data should have two columns, y_true and y_scores
        raise ValueError("The input file should have two columns, y_true and y_scores, please check the format of the input file")
    y_true = df['y_true']
    y_scores = df['y_scores']

    

    create_auc_plot(y_true, y_scores, args.save_path)

if __name__ == '__main__':
    main()
