def create_indexed_dataframe(index,actual,predicted):
  
    #Differenced the actual and predicted columns and add to a list to be put into a dataframe
    actual_diff    = actual.diff().tolist() 
    predicted_diff = predicted.diff().tolist() 
    
    #Put each remaining column into a list
    index          = index.tolist()
    actual         = actual.tolist()
    predicted      = predicted.tolist()

    #Create a dataframe with a dictionary of the remaining lists. 
    df = pd.DataFrame({'Index':  index,
                       'Actual':actual,
                       'Predicted':predicted,
                       'Actual Diff':actual_diff,
                       'Predicted Diff':predicted_diff}).set_index('Index')
    
    #Repace any infinite values with a NaN and then drop any rows with a nan value
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def add_residuals_normalization(df):
    
    import scipy.stats as st
    
    #Create the residuals column, normalized residuals, the Z Score, and Normalized Residuals by Percent for later analysis. 
    df['Residuals'] = df['Predicted'] - df['Actual']
    df['Normalized Residuals'] = (df['Residuals']-df['Residuals'].mean())/(df['Residuals'].max()-df['Residuals'].min())
    df['Residuals Zscore'] = (df['Normalized Residuals'] - df['Normalized Residuals'].mean())/df['Normalized Residuals'].std(ddof=0)
    df['Residuals Normalized Percent'] = df['Residuals Zscore'].apply(lambda x: st.norm.cdf(x))

    return df
    
def mean_forecast_error(actual,predicted):

    #Link:  https://en.wikipedia.org/wiki/Forecast_bias
    
    #A mean forecast error value other than zero suggests a tendency of the model to
    #over forecast (negative error) or under forecast (positive error).
    
    actual    = actual.tolist()
    predicted = predicted.tolist()

    mean_forecast_errors = [actual[i]-predicted[i] for i in range(len(actual))]
    bias = round(sum(mean_forecast_errors) * 1.0/len(actual),4)
    return bias

def mean_absolute_error(actual,predicted):
    
    #Link:  https://en.wikipedia.org/wiki/Mean_absolute_error
    
    from sklearn.metrics import mean_absolute_error
    
    actual    = actual.tolist()
    predicted = predicted.tolist()

    mae = round(mean_absolute_error(actual, predicted),4)
    
    return mae

def mean_squared_error(actual,predicted):
    
    #Link:  https://en.wikipedia.org/wiki/Mean_squared_error
        
    from sklearn.metrics import mean_squared_error
    
    actual    = actual.tolist()
    predicted = predicted.tolist()
    
    mse = round(mean_squared_error(actual,predicted),4)
    return mse

def mean_absolute_percentage_error(actual, predicted): 

    #Link:  https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    actual, predicted = np.array(actual), np.array(predicted)
    mape = round(np.mean(np.abs((actual - predicted) / actual)) * 100,4)
    
    return mape

def diagnostic_plot(df):
    
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    rcParams['figure.figsize'] = 18, 1
    plt.figtext(0,0,'Diagnostic Plots',fontsize = 24,fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    rcParams['figure.figsize'] = 18, 25
        
    fig = plt.figure()
    ax1 = plt.subplot2grid((8, 4), (0, 0), rowspan=2, colspan=4)
    ax2 = plt.subplot2grid((8, 4), (2, 0), rowspan=1, colspan=4)
    ax3 = plt.subplot2grid((8, 4), (3, 0), rowspan=1, colspan=3)
    ax4 = plt.subplot2grid((8, 4), (3, 3), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((8, 4), (4, 0), rowspan=1, colspan=3)
    ax6 = plt.subplot2grid((8, 4), (4, 3), rowspan=1, colspan=1)
    ax7 = plt.subplot2grid((8, 4), (5, 0), rowspan=3, colspan=2)
    ax8 = plt.subplot2grid((8, 4), (5, 2), rowspan=3, colspan=2)

    ax1.plot(df.index,df['Actual'], color = 'black', linewidth = 4, label = 'Actual')
    ax1.plot(df.index,df['Predicted'], color = 'skyblue', linewidth = 4, label = 'Predicted')
    ax1.set_title('Plot of the Predicted and Actual Values by the Order of the Values',fontsize = 14,fontweight='bold')
    ax1.legend(loc='best')

    ax2.fill_between( df.index,df['Actual Diff'], color="black", alpha=0.9,label = 'Actual Differenced')
    ax2.fill_between( df.index,df['Predicted Diff'], color="skyblue", alpha=0.4,label = 'Predicted Differenced')
    ax2.plot(df.index,df['Predicted Diff'], color="skyblue", linewidth = 4, alpha=0.6)
    ax2.set_title('First Differenced Actual and Predicted Values by the Order of the Values',fontsize = 14,fontweight='bold')
    ax2.legend(loc = 'best')

    ax3.scatter( df.index,df['Residuals'], color="black", s = 30,alpha=0.9, label = 'Residuals')
    ax3.axhline(y=0,color='skyblue',linestyle='--',linewidth = 2)
    ax3.set_title('Residuals vs. The Order of the Data',fontsize = 14,fontweight='bold')
    ax3.set_ylabel('Residuals',fontsize = 12)
    ax3.set_xlabel('Observation Order',fontsize = 12)
    ax3.legend(loc = 'best')
    
    bins = 10
    ax4.hist(df['Residuals'], color="black",bins = bins,orientation="horizontal")
    ax4.axhline(y=0,color='skyblue',linestyle='--',linewidth = 2)
    ax4.set_title('Histogram of the Residuals',fontsize = 14,fontweight='bold')
    ax4.set_ylabel('Residual',fontsize = 12)
    ax4.set_xlabel('Frequency',fontsize = 12)

    ax5.scatter( df['Predicted'],df['Residuals'], color="black", s = 30,alpha=0.9, label = 'Residuals')
    ax5.axhline(y=0,color='skyblue',linestyle='--',linewidth = 2)
    ax5.set_title('Residuals vs. The Fitted Values',fontsize = 14,fontweight='bold')
    ax5.set_ylabel('Residuals',fontsize = 12)
    ax5.set_xlabel('Fitted Values',fontsize = 12)
    ax5.legend(loc = 'best')

    ax6.scatter( df['Residuals'], df['Residuals Normalized Percent'],color="black", s = 30,alpha=0.9, label = 'Residuals')
    ax6.set_title('Normal Probability Plot of the Residuals',fontsize=14,fontweight='bold')
    ax6.set_ylabel('Percent',fontsize = 12)
    ax6.set_xlabel('Residuals',fontsize = 12)
    ax6.legend(loc = 'best')

    x1 = np.linspace(df['Predicted'].min(), df['Predicted'].max())
    ax7.scatter( df['Predicted'],df['Actual'], color="black", s = 30,alpha=0.9)
    ax7.plot(x1,x1, color = 'skyblue')
    ax7.set_title('Actual by Predicted Plot',fontsize=14,fontweight='bold')
    ax7.set_ylabel('Actual Values',fontsize = 12)
    ax7.set_xlabel('Predicted Values',fontsize = 12)

    x2 = np.linspace(df['Predicted Diff'].min(), df['Predicted Diff'].max())
    ax8.scatter( df['Predicted Diff'],df['Actual Diff'], color="black", s = 30,alpha=0.9)
    ax8.plot(x2,x2, color = 'skyblue')
    ax8.set_title('First Differences Actual by Predicted',fontsize=14,fontweight='bold')
    ax8.set_ylabel('Differenced Actual Values',fontsize = 12)
    ax8.set_xlabel('Differenced Predicted Values',fontsize = 12)

    plt.tight_layout()
    plt.show()
    
    return df

def plot_performance_measures(df):

    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    
    index = df.index
    actual = df['Actual']
    predicted = df['Predicted']
    
    
    bias = mean_forecast_error(actual,predicted)
    mae  = mean_absolute_error(actual,predicted)
    mse  = mean_squared_error(actual,predicted)
    mape = mean_absolute_percentage_error(actual,predicted)

    rcParams['figure.figsize'] = 18, 1

    plt.figtext(0,1,'Performance Measures',fontsize = 24,fontweight='bold')
    plt.figtext(0,.3, 'Mean Forecast Error (MFE) / Forecast Bias',fontsize = 22,fontweight='light')
    plt.figtext(0,.0, 'Mean Absolute Error (MAE)',fontsize = 22,fontweight='light')
    plt.figtext(0,-.3,'Mean Squared Error (MSE)'.format(mse),fontsize = 22,fontweight='light')
    plt.figtext(0,-.6,'Mean Absolute Percentage Error (MAPE)',fontsize = 22,fontweight='light')

    #Shift the locations of the scores in the plots to center the decimals!

    shift_bias = str(bias).find('.')*.01
    if bias <0:  shift_bias-= .004
    
    shift_mae  = str(mae).find('.')*.01
    if mae <0:  shift_mae-= .004
    
    shift_mse  = str(mse).find('.')*.01
    if mse <0:  shift_mse-= .004
    
    shift_mape = str(mape).find('.')*.01
    if mape <0:  shift_mape-= .004
    
    #Plot the locations of each statistics -- They should line up by decimal point
    plt.figtext(.5-shift_bias,.3, '{}'.format(bias),fontsize = 22,fontweight='light')
    plt.figtext(.5-shift_mae,.0, '{}'.format(mae) ,fontsize = 22,fontweight='light')
    plt.figtext(.5-shift_mse,-.3,'{}'.format(mse) ,fontsize = 22,fontweight='light')
    plt.figtext(.5-shift_mape,-.6,'{}'.format(mape),fontsize = 22,fontweight='light')


    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return 

def plot_ACF_PACF(df):

    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    
    #Plot the ACF and PACF figures for the non-differenced and differenced values
    rcParams['figure.figsize'] = 18, 1
    plt.figtext(0,0,'ACF and PACF Plots',fontsize = 24,fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    rcParams['figure.figsize'] = 18, 14

    fig = plt.figure()
    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((5, 4), (0, 2), rowspan=1, colspan=2)
    ax3 = plt.subplot2grid((5, 4), (1, 0), rowspan=1, colspan=2)
    ax4 = plt.subplot2grid((5, 4), (1, 2), rowspan=1, colspan=2)
    ax5 = plt.subplot2grid((5, 4), (2, 0), rowspan=1, colspan=2)
    ax6 = plt.subplot2grid((5, 4), (2, 2), rowspan=1, colspan=2)
    ax7 = plt.subplot2grid((5, 4), (3, 0), rowspan=1, colspan=2)
    ax8 = plt.subplot2grid((5, 4), (3, 2), rowspan=1, colspan=2)
    ax9 = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=2)
    ax10 = plt.subplot2grid((5, 4), (4, 2), rowspan=1, colspan=2)

    if round(len(df)) < 50:
        lags = round(len(df)*.75)
    else:
        lags = 50

    #ACF Plots
    plot_acf(df['Actual'],ax = ax1, use_vlines = True,title = 'ACF of the Actual Values',zero = False, lags = lags)
    plot_acf(df['Actual Diff'],ax = ax3, use_vlines = True,title = 'ACF of the Differenced Actual Values',zero = False, lags = lags)
    plot_acf(df['Predicted'],ax = ax5, use_vlines = True,title = 'ACF of the Predicted Values',zero = False, lags = lags)
    plot_acf(df['Predicted Diff'],ax = ax7, use_vlines = True,title = 'ACF of the Differenced Predicted Values',zero = False, lags = lags)
    plot_acf(df['Residuals'],ax = ax9, use_vlines = True,title = 'ACF of the Residuals',zero = False, lags = lags)

    #PACF Plots
    plot_pacf(df['Actual'],ax = ax2, use_vlines = True,title = 'PACF of the Actual Values',method='ywm',zero = False, lags = lags)
    plot_pacf(df['Actual Diff'],ax = ax4, use_vlines = True,title = 'PACF of the Differenced Actual Values',method='ywm', zero = False, lags = lags)
    plot_pacf(df['Predicted'],ax = ax6, use_vlines = True,title = 'PACF of the Predicted Values',zero = False, method='ywm' ,lags = lags)
    plot_pacf(df['Predicted Diff'],ax = ax8, use_vlines = True,title = 'PACF of the Differenced Predicted Values',method='ywm' ,zero = False, lags = lags)
    plot_pacf(df['Residuals'],ax = ax10, use_vlines = True,title = 'PACF of the Residuals',zero = False, method='ywm' ,lags = lags)

    plt.tight_layout()
    plt.show()
    return

def plot_table(df):
    
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    
    #Plot the title....
    rcParams['figure.figsize'] = 18, 1
    plt.figtext(0,0,'Table of Values',fontsize = 24,fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    #Plot the table of the values....
    df = round(df[['Actual','Predicted','Residuals','Actual Diff','Predicted Diff','Residuals Normalized Percent']],2)
    rcParams['figure.figsize'] = 18, len(df)/4
    plt.axis('off')
    plt.axis('tight')
    plt.table(cellText = df.values, rowLabels = df.index,colLabels=df.columns,loc='center',fontsize = 12)

    plt.show()
    return df


def time_series_diagnostic_main(index,actual,predicted, show_diagnostic_plots = True, show_performance_measures = True, show_ACF_PACF_plots = True, show_values = True):
    import pandas as pd
    import numpy as np
    from matplotlib import rcParams
    df = create_indexed_dataframe(index,actual,predicted)
    df = add_residuals_normalization(df)
    
    #Generate Plots
    if show_diagnostic_plots == True:
        diagnostic_plot(df)
    else: pass

    if show_performance_measures == True:
        plot_performance_measures(df)
    else: pass
    
    if show_ACF_PACF_plots == True:
        plot_ACF_PACF(df)
    else: pass
    
    if show_values == True:
        plot_table(df)
    else: pass

    
    return df
