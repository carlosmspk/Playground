import pandas as pd
from displayers import *
from file_settings import *

def read_csv (file_to_read : str):
    if not file_to_read.endswith('.csv'):
        file_to_read += '.csv'

    if file_to_read in file_settings_dict:
        file_settings = file_settings_dict[file_to_read]
    else:
        file_settings = default_settings

    try:
        sep = file_settings['sep']
    except KeyError:
        sep = ','

    try:
        displayer = file_settings['displayer']
    except KeyError:
        displayer = default_displayer

    df = pd.read_csv(file_to_read, sep=sep)
    return displayer, df