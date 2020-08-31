from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from ..plotting import plot_grid_results

# TODO: Expland the list of Models by accessing it from hyperparams_ml.pkl file
# TODO: Would need to change the xticklabels (an argument for custom labels)

DEFAULT_MODELS = [
    RidgeClassifier(),
    LogisticRegression(solver="liblinear"),  # liblinear is better for small datasets
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=1),
    SVC(),
    RandomForestClassifier(random_state=1),
    BaggingClassifier(random_state=1)
]

def print_best_param_score(search):
    print(f'Best param: {search.best_params_}', end="\n\n")
    print(f'Best score: {search.best_score_}')

def try_various_models(pipe, dfX, dfy, models='default', 
                       add_param_grid=None, search_type='grid', model_step='ml', plot=False, **kwargs):
    
    if models == 'default':
        param_grid = {model_step: DEFAULT_MODELS}
    else:
        param_grid = {model_step: models}
    
    if add_param_grid is not None:
        if type(add_param_grid) is dict:
            param_grid.update(add_param_grid)
        elif type(add_param_grid) is list:
            param_grid = [dict(pg, **param_grid) for pg in add_param_grid]
            
                
    if search_type == 'grid':
        search_cls = GridSearchCV
    elif search_type == 'rand':
        search_cls = RandomizedSearchCV
        
    search = search_cls(pipe, param_grid, **kwargs)
    search.fit(dfX, dfy)
    
    print_best_param_score(search)
    
    if plot:
        plot_grid_results(search, model_step)
        
    
    return search


def compare_param_effect_models(pipe, dfX, dfy, param_steps,
                                models='default', model_step='ml', alt='passthrough', comb=False, **kwargs):
    
    pipe_params = pipe.get_params()
    
    if type(alt) is not list:
        alt_opts = [alt for i in range(len(pipe_params))]
    else:
        alt_opts = alt
    
    if comb:
        add_param_grid = {ps: [pipe_params[ps], alt_opts[i]] 
                          for i, ps in enumerate(param_steps)}
    else:
        add_param_grid = [{ps: [pipe_params[ps], alt_opts[i]]} 
                          for i, ps in enumerate(param_steps)]
        
        
    return try_various_models(pipe, dfX, dfy, models, add_param_grid, model_step=model_step, **kwargs)




    