from displayers import *

# ADD NEW FILE NAMES AND SETTINGS HERE AND THEN ADD THEM TO THE DICTIONARY FURTHER BELOW

example_custom_settings = {
    'sep' : ';',
    'displayer' : example_custom_displayer
}

nature_inspired_algorithms = {
    'sep' : ';',
    'displayer' : nature_algos_displayer
}

# ADD THE PREVIOUSLY CREATED DICTIONARY TO THIS DICTIONARY
file_settings_dict = {
    'example_csv.csv' : example_custom_settings,
    'nature_inspired_algorithms.csv': nature_inspired_algorithms
}

# Default settings that are used whenever a previous field in 'file_settings_dict' is incomplete or the .csv file doesn't show up in 'file_settings_dict'
default_settings = {
    'sep' : ',',
    'displayer' : default_displayer
}