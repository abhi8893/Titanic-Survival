import yaml
import os

HOME = os.environ.get('HOME')
APP_NAME = 'titansurv'
APP_CONFIG_FILE = os.path.join(HOME, f'.{APP_NAME}')

def set_config_file_location(config_file):
    with open(APP_CONFIG_FILE, "w") as f:
        yaml.dump({'CONFIG_FILE': config_file}, f, default_flow_style=False)

def get_config_file_location():
    with open(APP_CONFIG_FILE, "r") as f:
        config_file = yaml.safe_load(f)['CONFIG_FILE']

    return config_file

def get_raw_config():
    config_file = get_config_file_location()
        
    with open(config_file, "r") as f:
        config_params = yaml.safe_load(f)

    return config_params


def get_config():
    config_params = get_raw_config()
    IS_RELATIVE = config_params.pop('IS_RELATIVE')
    PROJECT_DIR = config_params['PROJECT_DIR']
    config_params_abs_path = {k: os.path.join(PROJECT_DIR, v) 
                              for k, v in IS_RELATIVE.items()}

    config_params.update(config_params_abs_path)

    return config_params
    
    

