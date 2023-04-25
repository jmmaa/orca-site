import streamlit as st
import pandas as pd

from model import model
from xgboost import DMatrix

from datetime import date, timedelta
import typing as t

value = t.cast(date, st.date_input(label="Pick a starting prediction date"))


years: list[int] = []
months: list[int] = []
days: list[int] = []

for i in range(7):
    date_tuple = (value + timedelta(days=i)).timetuple()
    year = date_tuple[0]
    month = date_tuple[1]
    day = date_tuple[2]

    years.append(year)
    months.append(month)
    days.append(day)

dates_to_predict = pd.DataFrame({"day": days, "month": months, "year": years})


predictions = pd.DataFrame(model.predict(DMatrix(dates_to_predict)))
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
predictions["date"] = pd.to_datetime(
    dates_to_predict[["year", "month", "day"]]
).dt.strftime("%Y-%m-%d")

# set index to date
predictions = predictions.set_index("date")

st.line_chart(predictions["mean water level"])
st.line_chart(predictions["mean sea level pressure"])
st.line_chart(predictions[["max temperature", "min temperature", "mean temperature"]])
st.line_chart(predictions[["mean relative humidity"]])
st.line_chart(predictions[["mean wind speed"]])
