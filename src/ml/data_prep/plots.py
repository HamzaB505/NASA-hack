import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.ml import logger


def plot_data_distributions(df, figsize=(20, 15), cols_per_row=2):

    # Get number of columns and split into smaller groups for better visibility
    n_cols = len(df.columns)
    cols_per_plot = 6  # Much smaller groups to avoid crushing plots
    
    logger.info(f"Plotting data distributions for {n_cols} columns, split into groups of {cols_per_plot}")
    
    # Split columns into groups
    column_groups = []
    for i in range(0, n_cols, cols_per_plot):
        column_groups.append(df.columns[i:i + cols_per_plot])
    
    logger.info(f"Created {len(column_groups)} column groups for plotting")
    
    # Plot each group
    for idx, cols_group in enumerate(column_groups):
        title = (f"Data Distribution Analysis - Part {idx + 1} of "
                 f"{len(column_groups)}")
        logger.info(f"Plotting group {idx + 1}/{len(column_groups)}: {list(cols_group)}")
        _plot_column_group(df[cols_group], title, figsize, cols_per_row)


def _plot_column_group(df_subset, title, figsize, cols_per_row):
    """
    Helper function to plot a subset of columns with much better spacing
    """
    n_cols = len(df_subset.columns)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row
    
    logger.debug(f"Creating {n_rows}x{cols_per_row} subplot grid for {n_cols} columns")
    
    # Create figure with larger size and better spacing
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
    fig.suptitle(title, fontsize=18, y=0.98)
    
    # Flatten axes array for easier indexing
    if n_rows == 1 and cols_per_row == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif cols_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    # Plot each column with much better formatting
    for i, col in enumerate(df_subset.columns):
        ax = axes[i]
        
        # Get column data type
        dtype = df_subset[col].dtype
        
        if pd.api.types.is_float_dtype(dtype):
            # Float columns: histogram with statistics
            data = df_subset[col].dropna()
            if len(data) > 0:
                logger.debug(f"Plotting float column '{col}' with {len(data)} non-null values")
                # Use fewer bins for cleaner look
                n_bins = min(30, max(10, int(np.sqrt(len(data)))))
                ax.hist(data, bins=n_bins, alpha=0.7, edgecolor='black',
                        linewidth=0.8, color='skyblue')
                
                # Set x-axis limits to be close to min and max values for
                # better visibility
                data_min = data.min()
                data_max = data.max()
                data_range = data_max - data_min
                
                # Add small padding (5% of range) to avoid cutting off bars
                padding = data_range * 0.05 if data_range > 0 else 0.1
                ax.set_xlim(data_min - padding, data_max + padding)
                
                # Add statistics text with better formatting
                mean_val = data.mean()
                std_val = data.std()
                median_val = data.median()
                
                stats_text = (f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\n'
                              f'Median: {median_val:.4f}\nCount: {len(data)}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.9, edgecolor='gray'))
                
                ax.set_title(f'{col}\n(Float Distribution)', fontsize=12,
                             pad=15, weight='bold')
                ax.set_xlabel(col, fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.grid(True, alpha=0.3)
            
        elif pd.api.types.is_integer_dtype(dtype):
            # Integer columns: bar plot of value counts
            value_counts = df_subset[col].value_counts().sort_index()
            
            logger.debug(f"Plotting integer column '{col}' with {len(value_counts)} unique values")
            
            # Limit number of bars if too many unique values
            if len(value_counts) > 15:
                value_counts = value_counts.head(15)
                title_suffix = (f'(Integer - Top 15 of '
                                f'{df_subset[col].nunique()} values)')
            else:
                title_suffix = f'(Integer - All {len(value_counts)} values)'
            
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                          alpha=0.7, color='lightcoral')
            ax.set_title(f'{col}\n{title_suffix}', fontsize=12, pad=15,
                         weight='bold')
            ax.set_xlabel('Values', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            
            # Set x-tick labels with better spacing
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right',
                               fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars if reasonable number
            if len(value_counts) <= 8:
                for bar, count in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(value_counts.values)*0.02,
                            str(count), ha='center', va='bottom',
                            fontsize=10, weight='bold')
            
        else:
            # Object columns: convert to category and bar plot
            cat_data = df_subset[col].astype('category')
            value_counts = cat_data.value_counts()
            
            logger.debug(f"Plotting object column '{col}' with {len(value_counts)} categories")
            
            # Limit to top categories if too many
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
                title_suffix = (f'(Object - Top 10 of '
                                f'{len(cat_data.cat.categories)} categories)')
            else:
                title_suffix = f'(Object - All {len(value_counts)} categories)'
            
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                          alpha=0.7, color='lightgreen')
            ax.set_title(f'{col}\n{title_suffix}', fontsize=12, pad=15,
                         weight='bold')
            ax.set_xlabel('Categories', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            
            # Set x-tick labels with better rotation and spacing
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right',
                               fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars if reasonable number
            if len(value_counts) <= 8:
                for bar, count in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(value_counts.values)*0.02,
                            str(count), ha='center', va='bottom',
                            fontsize=10, weight='bold')
        
        # Make tick labels more readable
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add border around each subplot
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    # Hide unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout with much more generous spacing
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.92,
                        wspace=0.4, hspace=0.6)
    plt.show()


# Plot distributions for X with enhanced visualization split into multiple
# well-spaced parts
# plot_data_distributions(X)
