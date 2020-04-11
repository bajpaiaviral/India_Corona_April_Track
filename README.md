# India_Corona_April_Track

Daily CoronaVirus Cases and Death data was taken from us_covid19_daily.csv csv file taken from Kaggle Country-Wise Dataset.

And then an LSTM neural network was trained on the time series data after the data was wrangled for the LSTM input format.

Corona Virus cases in the start of April Month for further prediction were extracted from covid_19_india.csv . Then in a loop 
predictions for the next day were calculated cyclically .

The predictions of Deaths and Positive Results were finally visualized on a Bar Chart which predicted around 82000 cases And 6000 deaths by 30th April .
