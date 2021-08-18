def default_displayer (df) -> None:
    print(df.head())

def example_custom_displayer (df) -> None:
    print(df.head(n=10))
    
def nature_algos_displayer (df) -> None:
    for index, row in df.iterrows():
        if '' in row['Algorithm_name']:
            if row['Abbr'] == ' ':
                print(index, '\t', row['Algorithm_name'], sep='')
            else:
                print(index, '\t', row['Algorithm_name'], ' (', row['Abbr'], ')', sep='')