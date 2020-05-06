import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: plot top n results for x
# TODO: plot top n results for a specific ML model
# TODO: plot top n for a specific param
def plot_grid_results(grid, x, kind='score'):
    
    '''Plot the results of a GridSearchCV object
    
    Parameters
    ----------
    grid: GridSearchCV object
        A fitted GridSearchCV object
        
    x: str
        The parameter to plot the barplot for
        
    kind: str
        Plot the mean 'scores' or mean 'time' of fitting each combination
    
    '''
    
    res = pd.DataFrame(grid.cv_results_)
    
    if kind == 'score':
        y = 'mean_test_score'
        ascending = False
        better = 'higher'
    elif kind == 'time':
        y = 'mean_score_time'
        ascending = True
        better = 'lower'
        
    res.sort_values(y, ascending=ascending, inplace=True)

        
    g = sns.barplot(f'param_{x}', y, data=res)
    xticklabels = res[f'param_{x}'].apply(lambda m: m.__class__.__name__)
    g.set_xticklabels(xticklabels, rotation=30)
    g.set_xlabel(f'{better} is better',  {'weight': 'bold', 'size': 20}, labelpad=20)
    g.set_ylabel(f'mean {kind}', labelpad=20)
    
    g.figure.set_size_inches(12, 8)
    g.set_title(f'GridSearchCV crossvalidated {kind}', {'weight': 'bold', 'size': 20})
    
    return g



def plot_count(series, dropna=False):
    val_cnts = series.value_counts(dropna=dropna)
    plt.figure(figsize=(20, 1))
    
    if not dropna:
        try:
            series = series.fillna('NaN')
        except ValueError: # fill value must be in category error
            pass
        
    g = sns.countplot(y=series)
    
    print("The frequency of each category:\n", 
          val_cnts, "\n", sep="")
    print("The proportion of each category:\n", 
          val_cnts/series.size, "\n", sep="")
    
    return g



def plot_prob(x, y, data=None):
    if data is None:
        data = pd.concat([x, y], axis=1)
        x, y = data.columns
        data = data.dropna()
    
    
    g = sns.catplot(x, y, kind='bar', data=data)
    g.despine(left=True)
    g.set_ylabels(f'{y} probability')
    
    return g


def plot_missprop(df):
    miss_prop = df.isna().mean()
    fig, ax = plt.subplots(figsize=(12, 8))
    miss_prop.plot(kind="bar", ax=ax)
    ax.set_title("Missing Proportions", fontdict=dict(size=20, weight="bold"))
    
    return(miss_prop)