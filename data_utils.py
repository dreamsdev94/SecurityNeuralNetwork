import pandas as pd

def create_dataset(filename):
    df = pd.read_csv(filename)

    sets = df.drop(columns=['label','type'])
    label_targets = df['label'].to_numpy()

    cols_to_delete = list()

    for col in sets.columns:
        min_value = sets[col].min()
        max_value = sets[col].max()

        if min_value == max_value:
            cols_to_delete.append(col)
            continue

        sets[col] = -1 + (sets[col] - min_value) * 2 / (max_value - min_value)

    sets = sets.drop(columns=cols_to_delete)

    return sets.to_numpy(), label_targets
