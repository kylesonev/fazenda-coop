import pandas as pd

def spliting_train_test(df: pd.DataFrame, date_column: str) -> pd.Series:
    """
    Splits a data frame for train and test
    Train: 80%
    Test: 20%
    """
    df = df.drop(columns=[date_column])
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy() 
    test = df.iloc[split_idx:].copy()

    return train, test


def preparing_prophet(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Prepares a dataframe for Prophet model
    """
    if "date" in df.columns:
        df_prophet = df[['date', target]].rename(columns={'date': 'ds', target: 'y'}).copy()
    else:
        df_prophet = df.reset_index()[['date', target]].rename(columns={'date': 'ds', target: 'y'}).copy()
    return df_prophet


def preparing_arima(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Prepares X and y for arima
    """
    X = df.drop(columns=[target])
    y = df[target]

    return X, y



## Para séries temporais (como ARIMA)
# train, test = spliting_train_test(df)
# X_train, y_train = preparing_arima(train, target_column='target')
# X_test, y_test = preparing_arima(test, target_column='target')