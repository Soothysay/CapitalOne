import numpy as np
import pandas as pd 
from tabulate import tabulate
from prettytable import PrettyTable
from termcolor import cprint
import statistics

def read_data(path: str) -> pd.DataFrame:
# Simple function to read a json file into a pandas dataframe
    df = pd.read_json(path, lines=True)
    df.replace('', np.nan, inplace=True)
    return df

def save_data(df: pd.DataFrame, path: str) -> None:
# Simple function to save a pandas dataframe to a csv file
    df.to_csv(path, index=True)

def basic_stats(df,fets):
    cprint('SUMMARY STATISTICS : ', attrs=['bold'])
    print()
    #fets=[input("Enter the column name")]
    features=fets
    #features=(['5% quantile','95% quanrtile','skewness','kurtosis','variance','standard deviation'])
    basic_stat=pd.DataFrame(columns=['features'])
    basic_stat.loc['MAX']= df.max()
    basic_stat.loc['MIN']= df.min()
    basic_stat.loc['Range']=df[fets].max()-df[fets].min()
    basic_stat.loc['skewness']= df[fets].skew()
    basic_stat.loc['kurtosis']= df[fets].kurtosis()
    basic_stat.loc['variance']= df[fets].var()
    basic_stat.loc['standard deviation']= df[fets].std()
    d=basic_stat
    display(d)

def display_value_counts(df, column_name):

    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return None

    # Calculate value counts
    value_counts_series = df[column_name].value_counts()

    # Convert to DataFrame
    value_counts_df = pd.DataFrame({column_name: value_counts_series.index, 'Count': value_counts_series.values})

    return value_counts_df

def bdist_plot(df,ch):
    import plotly.figure_factory as ff
    hist_data=[df[ch]]
    group_labels = [ch]
    fig = ff.create_distplot(hist_data,group_labels,colors=['#32E0C4'])  #Change colours with palette
                         #title='Distplot',
                         #template='simple_white')marker=dict(color='#32E0C4')
    fig['layout'].update(title='Distribution Plot: {}'.format(ch),title_x=0.5)
    #fig.update_layout(title_text="<b>Distribution Plot<b> : {} ".format(fet), title_x=0.5)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)','paper_bgcolor' : 'rgba(0,0,0,0)'})
    #fig.update_xaxes(title_text='<b>' + fet +'<b>')
    #fig.update_yaxes(title_text='<b>COUNT<b>')
    fig.show()