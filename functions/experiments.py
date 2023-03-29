import os
from dotenv import dotenv_values
import yaml
from yaml.loader import SafeLoader

class Experiments:
    
    def __init__(self):
        self.env = dotenv_values('.env')
    
    def get_all_experiments(self):
        PATH = self.env.get('PATH_MLRUNS')
        experiments = []
        experiments_id = list(os.walk(PATH))
        experiments_name = experiments_id[0][1]
        experiments_name.remove('.trash')
        experiments_name.remove('models')

        for experiment in experiments_name:
            current_path = f'{PATH}\\{experiment}\\meta.yaml'
            with open(current_path) as f:
                try:
                    data = yaml.load(f, Loader=SafeLoader)
                    info_run = {
                        'artifact_location': data['artifact_location'],
                        'creation_time': data['creation_time'],
                        'experiment_id': data['experiment_id'],
                        'last_update_time': data['last_update_time'],
                        'lifecycle_stage': data['lifecycle_stage'],
                        'name': data['name']
                    }
                    experiments.append(info_run)
                except OSError as err:
                    print(f'Error when trying to open the directory: {err}')
                    
        return experiments