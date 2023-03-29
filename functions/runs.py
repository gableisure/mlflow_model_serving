import os
from dotenv import dotenv_values
import yaml
from yaml.loader import SafeLoader

class Runs:
    
    def __init__(self):
        self.env = dotenv_values('.env')