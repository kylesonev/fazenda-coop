from prophet import Prophet
from pmdarima import auto_arima
import pandas as pd


def training_prophet(df_prophet: pd.DataFrame) -> Prophet:
    """
    Trains a Prophet model based on a DataFrame
    """
    model = Prophet()
    model.fit(df_prophet)
    return model


def forecasting_prophet(model: Prophet) -> pd.DataFrame:
    """
    Generates a forecast for the next 12 months
    """
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    return forecast


def training_auto_arima(y_train: pd.Series, X_train: pd.Series = None) -> auto_arima:
    """
    Trains an auto_arima model
    """
    model = auto_arima(
        y=y_train,
        exogenous=X_train if X_train is not None else None,
        seasonal=True,
        m=12,
        trace=False,
        error_action="ignore",
        suppress_warnings=True
    )
    return model