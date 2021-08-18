from file_settings import *
from csv_reader import read_csv

file_to_read = 'example_csv'

displayer, df = read_csv(file_to_read)
displayer(df)