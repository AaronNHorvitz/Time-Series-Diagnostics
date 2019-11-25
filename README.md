# Time-Series-Diagnostics
This code provides a series of diagnostic plots for time series analysis. These include ACF and PACF plots, predicted vs actual, residual plots, differenced residual plots, and many more.

Here is the execution of the code:

time_series_diagnostic_main(index,actual,predicted, show_diagnostic_plots = True, show_performance_measures = True, show_ACF_PACF_plots = True, show_values = True)

Here are is a description of the elements in the code. 

'index'     is the ordered index of values. This could be daily, monthly, annual values, or just an ordered set of values in series.
'actual'    are the the actual values in the series
'predicted' are the predicted values by the model

'show_diagnostic_plots = True'     displays the diagnostic plots
'show_performance_measures = True' displays the model performance statistics
'show_ACF_PACF_plots = True'       displays the ACF and PACF plots
'show_values = True'               displays the actual, predicted, and residual values
