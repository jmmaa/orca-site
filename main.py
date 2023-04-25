import streamlit as st
import pandas as pd

from model import model
from xgboost import DMatrix

from datetime import date, timedelta
import typing as t


def generate_dates(from_: date, to_: int) -> pd.DataFrame:
    years: list[int] = []
    months: list[int] = []
    days: list[int] = []

    for i in range(to_):
        date_tuple = (from_ + timedelta(days=i)).timetuple()
        year = date_tuple[0]
        month = date_tuple[1]
        day = date_tuple[2]

        years.append(year)
        months.append(month)
        days.append(day)

    return pd.DataFrame({"day": days, "month": months, "year": years})


def generate_predictions(to_pred: pd.DataFrame):
    # convert to xgboost-compatible data format
    matrix = DMatrix(to_pred)

    # predict
    predictions = model.predict(matrix)

    # convert to df and rename columns

    predictions = pd.DataFrame(predictions)
    predictions.rename(
        {
            0: "mean water level",
            1: "mean sea level pressure",
            2: "max temperature",
            3: "min temperature",
            4: "mean temperature",
            5: "mean relative humidity",
            6: "mean wind speed",
        },
        axis=1,
        inplace=True,
    )

    # convert date values to string
    predictions["date"] = pd.to_datetime(to_pred[["year", "month", "day"]]).dt.strftime(
        "%Y-%m-%d"
    )

    # set index to date
    predictions = predictions.set_index("date")

    return predictions


def main():
    st.set_page_config(page_title="ORCA", page_icon="./assets/orca.png")

    st.write(
        """
    # O.R.C.A Model

    Welcome to O.R.C.A Model playground! It is a machine learning model trained with
    [PAGASA](https://www.pagasa.dost.gov.ph/) and [NAMRIA](https://www.namria.gov.ph/) 
    datasets from 2021. This model is used for forecasting weather at coastal areas in
    Davao City.

    
    ## How to use

    To use the model, just simply put the starting date and the span of days you want
    to forecast.

    **Important Note**: having span of days more than a week and a starting date further
    beyond **2022/01/01** will lead to significantly innacurate results.
    """
    )

    inp_col1, inp_col2 = st.columns(2)

    starting_date = t.cast(date, inp_col1.date_input(label="starting date"))
    span = inp_col2.number_input(label="span of days", min_value=1)

    dates = generate_dates(from_=t.cast(date, starting_date), to_=int(span))
    preds = generate_predictions(dates)

    st.divider()
    st.write(
        f"""
        Mean Water Level from `{starting_date}` to 
        `{starting_date + timedelta(span)}` measured in centimeters (cm)
        """
    )
    st.line_chart(preds["mean water level"])
    st.divider()
    st.write(
        f"""
        Mean Sea Level Pressure from `{starting_date}` to 
        `{starting_date + timedelta(span)}` measured in millibars (mbar)
        """
    )
    st.line_chart(preds["mean sea level pressure"])
    st.divider()
    st.write(
        f"""
       Temperature from `{starting_date}` to 
        `{starting_date + timedelta(span)}` measured in degrees celsius
        """
    )
    st.line_chart(preds[["max temperature", "min temperature", "mean temperature"]])
    st.divider()
    st.write(
        f"""
        Mean Relative Humidity from `{starting_date}` to 
        `{starting_date + timedelta(span)}` measured in percent
        """
    )
    st.line_chart(preds["mean relative humidity"])
    st.divider()
    st.write(
        f"""
        Mean Wind Speed from `{starting_date}` to 
        `{starting_date + timedelta(span)}` measured in meters per second
        """
    )
    st.line_chart(preds["mean wind speed"])
    st.divider()


if __name__ == "__main__":
    main()
