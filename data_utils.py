import pandas as pd

def create_dataset(filename):
    df = pd.read_csv(filename)

    sets = df.drop(columns=['label','type'])
