import matplotlib.pyplot as plt
import numpy as np
import os

def plot_function_approximation(X_test, y_test, y_pred, title, filename):
    """
    Plots the ground truth sine function and the network approximation.
    """
    plt.figure(figsize=(10, 6))
    
    # Sort for clean plotting
    sort_idx = np.argsort(X_test.flatten())
    X_sorted = X_test.flatten()[sort_idx]
    y_test_sorted = y_test.flatten()[sort_idx]
    y_pred_sorted = y_pred.flatten()[sort_idx]
    
    plt.plot(X_sorted, y_test_sorted, label='Ground Truth (sin(x))', color='blue', linewidth=2)
    plt.plot(X_sorted, y_pred_sorted, label='Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    plt.close()

def plot_oinfo_gradients(layer_gradients, layer_oinfos, filename_prefix):
    """
    Plots the local O-information gradients for each layer.
    """
    n_layers = len(layer_gradients)
    
    for i in range(n_layers):
        grads = layer_gradients[i]
        oinfo = layer_oinfos[i]
        
        plt.figure(figsize=(12, 6))
        
        # Color bars: Red for positive (redundant), Blue for negative (synergistic)
        colors = ['red' if g > 0 else 'blue' for g in grads]
        
        plt.bar(range(len(grads)), grads, color=colors)
        
        plt.title(f'Layer {i+1} O-Info Gradients (Total O-Info: {oinfo:.4f})')
        plt.xlabel('Neuron Index')
        plt.ylabel('Local O-Info Gradient')
        plt.axhline(0, color='black', linewidth=0.5)
        
        # Highlight redundant neurons (positive gradient)
        plt.text(0.02, 0.95, 'Positive = Redundant', transform=plt.gca().transAxes, color='red', fontweight='bold')
        plt.text(0.02, 0.90, 'Negative = Synergistic', transform=plt.gca().transAxes, color='blue', fontweight='bold')
        
        plt.savefig(f"{filename_prefix}_layer_{i+1}.png")
        plt.close()

def plot_comparison(metrics_pre, metrics_post, filename):
    """
    Plots a comparison of metrics before and after pruning.
    metrics: dict with keys 'mse', 'oinfo_total'
    """
    labels = ['MSE', 'Total O-Info']
    pre_values = [metrics_pre['mse'], metrics_pre['oinfo_total']]
    post_values = [metrics_post['mse'], metrics_post['oinfo_total']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, pre_values, width, label='Pre-Pruning')
    rects2 = plt.bar(x + width/2, post_values, width, label='Post-Pruning')
    
    plt.ylabel('Value')
    plt.title('Comparison Pre vs Post Pruning')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.savefig(filename)
    plt.close()

def plot_model_size_comparison(sizes_pre, sizes_post, filename):
    """
    Plots comparison of number of neurons and parameters.
    sizes_pre: dict {'neurons': int, 'params': int}
    sizes_post: dict {'neurons': int, 'params': int}
    """
    labels = ['Neurons', 'Parameters']
    pre_values = [sizes_pre['neurons'], sizes_pre['params']]
    post_values = [sizes_post['neurons'], sizes_post['params']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # We use two y-axes because params are much larger than neurons
    ax2 = ax1.twinx()
    
    # Plot Neurons on left axis
    p1 = ax1.bar(x[0] - width/2, pre_values[0], width, label='Pre-Pruning (Neurons)', color='tab:blue')
    p2 = ax1.bar(x[0] + width/2, post_values[0], width, label='Post-Pruning (Neurons)', color='tab:orange')
    
    # Plot Params on right axis
    p3 = ax2.bar(x[1] - width/2, pre_values[1], width, label='Pre-Pruning (Params)', color='tab:green')
    p4 = ax2.bar(x[1] + width/2, post_values[1], width, label='Post-Pruning (Params)', color='tab:red')
    
    ax1.set_ylabel('Number of Neurons')
    ax2.set_ylabel('Number of Parameters')
    ax1.set_title('Model Size Comparison')
    
    # Set x-ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    # Combine legends
    lines = [p1, p2, p3, p4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center')
    
    # Add labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(p1, ax1)
    autolabel(p2, ax1)
    autolabel(p3, ax2)
    autolabel(p4, ax2)
    
    plt.savefig(filename)
    plt.close()
