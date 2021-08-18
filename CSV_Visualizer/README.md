# Inserting .csv Files

Before reading a file you need to:

1. Insert a reader function in 'displayers.py'
2. Insert the file structure (seperator and displayer function to use) in 'file_settings.py' associated to the file name

# Running the Reader

In order to run the reader just go into 'main.py', change the 'file_to_read' variable to the desired .csv file (with or without .csv included) and run the script. If no configuration was defined, then comma seperators will be assumed, and the first 5 rows will be printed. The same happens for each field that wasn't defined, even if they were included in 'file_settings.py'.