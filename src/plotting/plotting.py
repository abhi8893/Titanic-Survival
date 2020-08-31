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


def boxplot_compare(df1, df2, clmns: list, nrow, ncol, keys=None, figsize=None, **kwargs):
    
    fig, axn = plt.subplots(nrow, ncol, figsize=figsize)
    
    if len(clmns) == 1:
        axn = [axn]
        
    if keys is None:
        keys = ['df1', 'df2']
    
        
    df_all = pd.concat([df1[clmns], df2[clmns]], keys=keys, 
                       names=['dataset'], axis=0).droplevel(1).reset_index()
    

    for ax, clmn in zip(axn, clmns):
        sns.boxplot(x='dataset', y=clmn, data=df_all, ax=ax, **kwargs)
        ax.set_title(clmn, fontdict={'weight': 'bold', 'size': 20})
        
        
    title = 'Comparing {} and {} datasets'.format(*keys)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    
    return fig, axn



# BUG: df2 gets plotted on the left and df1 on the right. Should be opposite.
def barplot_compare(df1, df2, clmns: list, nrow, ncol, keys=None, figsize=None, **kwargs):
    fig, axn = plt.subplots(nrow, ncol, figsize=figsize)
    
    if len(clmns) == 1:
        axn = [axn]
        
    if keys is None:
        keys = ['df1', 'df2']
    
        
    df_all = pd.concat([df1[clmns], df2[clmns]], keys=keys, 
                       names=['dataset'], axis=0).droplevel(1).reset_index()
    

    for ax, clmn in zip(axn, clmns):
        (df_all
         .groupby('dataset')[clmn]
         .value_counts(normalize=True)
         .rename('proportion')
         .reset_index()
         .pipe((sns.barplot, 'data'), x='dataset', y='proportion', hue=clmn, ax=ax))
        
        ax.set_title(clmn, fontdict={'weight': 'bold', 'size': 20})
        
        
    title = 'Comparing {} and {} datasets'.format(*keys)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    
    return fig, axn
    

def plot_param_effect_models(search, param_step, model_step='ml', labels=None):
    res = pd.DataFrame(search.cv_results_)

    res['ml_name'] = res[f'param_{model_step}'].apply(lambda x: x.__class__.__name__)
        
    param_col = f'param_{param_step}'
    valid_idx = ~res[param_col].isna()
    valid_res = res.loc[valid_idx, :]
    index_best = valid_res.index[valid_res.rank_test_score.argmin()]

    best_param = valid_res.loc[index_best, param_col]

    order = (res.loc[res[f'param_{param_step}'] == best_param, :]
             .sort_values('rank_test_score')['ml_name'].tolist())
    
    plt.figure(figsize=(12, 8))
    g = sns.barplot(x='ml_name', y='mean_test_score', hue=f'param_{param_step}', order=order,
                data=res)

    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if labels is not None:
        for t, l in zip(g.legend_.texts, labels): t.set_text(l)

    
    return g


