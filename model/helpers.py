from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mape(y_true: Union[list, np.array, pd.Series], y_pred: Union[list, np.array, pd.Series]) -> float:   
    """Calculate Mean Absolute Percentage Error (MAPE)    
    Implements formula from https://en.wikipedia.org/wiki/Mean_absolute_percentage_error    
     
    Args:
    y_true (Union[list, np.array, pd.Series]): Vector of observed response values        
    y_pred (Union[list, np.array, pd.Series]): Vector of predicted response values    
        
    Returns:
        float: calculated MAPE value   
    """    
    y_true, y_pred = np.array(y_true), np.array(y_pred)    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     
def mad(y_true: Union[list, np.array, pd.Series], y_pred: Union[list, np.array, pd.Series]) -> float:    
    """Calculate Mean Absolute Deviation (MAD)    
    Implements formula from https://en.wikipedia.org/wiki/Mean_absolute_error.    
    Also known as Mean Absolute Error (MAE).    
    
    Args:        
        y_true (Union[list, np.array, pd.Series]): Vector of observed response values        
        y_pred (Union[list, np.array, pd.Series]): Vector of predicted response values    
    
    Returns:        
        float: calculated MAD value    
    """    
    y_true, y_pred = np.array(y_true), np.array(y_pred)    
    return np.mean(np.abs(y_true - y_pred))
    
def fitted_vs_actuals_plot(fitted: Union[list, np.array, pd.Series], actuals: Union[list, np.array, pd.Series], category: str,):    
    fitted, actuals = np.array(fitted), np.array(actuals)    
    fig = plt.figure(figsize=(8, 8))    
    min_x, max_x = np.min(actuals), np.max(actuals)    
    min_y, max_y = np.min(fitted), np.max(fitted)    
    plt.scatter(x=actuals, y=fitted)    
    plt.plot([min_x, max_x], [min_y, max_y], ls="--", c="red")    
    plt.title(f"{category}: fitted vs actuals")    
    plt.xlabel("Actuals")    
    plt.ylabel("Fitted")    
    return fig
    
def residuals_plot(residuals: Union[list, np.array, pd.Series], category: str,):    
    x_vals, residuals = np.arange(residuals.shape[0]), np.array(residuals)    
    left, width = 0.1, 0.65    
    bottom, height = 0.1, 0.8    
    spacing = 0.005    
    fig = plt.figure(figsize=(8, 8))    
    ax = fig.add_axes([left, bottom, width, height])    
    hist_ax = fig.add_axes([left + width + spacing, bottom, 0.2, height], sharey=ax)    
    hist_ax.tick_params(axis="y", labelleft=False)    
    
    ax.scatter(x=x_vals, y=residuals)    
    fig.suptitle(f"{category}: residuals")    
    ax.set_xlabel("")    
    ax.set_ylabel("Residuals")    
    ax.plot(x_vals, np.zeros_like(x_vals), ls="--", c="red")    
    
    hist_ax.hist(residuals, bins="auto", orientation="horizontal")    
    return fig



###### IMPORTANT #######
# Helpers.py contains some custom functions for plotting and metric calculation. 
# These functions are imported by model.py.
# model.py contains the modeling code. 
# While the model itself isn't the main focus of this course, pay particular attention to usage of the run variable. 
# This variable is an instance of the AzureML SDK's Run class. 
# A run represents a single trial of an experiment and is used to monitor the asynchronous execution of a trial, log metrics, and store the output of the trial. 
# Here, we use the run variable to log metrics and artifacts through the run.log(), run.log_row(), run.log_residuals(), and run.log_image() methods.
