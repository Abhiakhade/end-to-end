import pandas as pd


def train_test_split(df):
    """
    Leave-one-out split:
    - Last interaction per user → test
    - Rest → train
    """

    train_list = []
    test_list = []

    grouped = df.groupby('user')

    for user, group in grouped:
        group = group.sample(frac=1, random_state=42)

        if len(group) < 2:
            continue

        test = group.iloc[-1:]
        train = group.iloc[:-1]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df