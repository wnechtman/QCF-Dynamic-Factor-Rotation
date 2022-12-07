# Support Functions for downloading data

import pandas as pd
import warnings


# Get fama french data
def fama_french(from_web=True, since_y=None) -> pd.DataFrame:
    # From web
    if from_web:
        ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
        data = pd.read_csv(ff5_url, skiprows=2).iloc[:-60, :]
    # Or from local file
    if not from_web:
        data = pd.read_csv("/Users/wyattnechtman/Documents/TECH/Fall2022/MGT6785/Project/Data/F-F_Research_Data_5_Factors_2x3.csv", skiprows=12).iloc[:-99, :]

    data.columns = ["Date", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    data["Date"] = pd.to_datetime(data["Date"], format="%Y%m")
    data.set_index("Date", inplace=True)
    data = data.astype(float)
    data /= 100

    # Add momentum
    if from_web:
        mom = pd.read_csv("http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip", skiprows=12).iloc[:-99, :]
    if not from_web:
        mom = pd.read_csv("/Users/wyattnechtman/Documents/TECH/Fall2022/MGT6785/Project/Data/F-F_Momentum_Factor.CSV", skiprows=12).iloc[:-99, :]
    mom.columns = ["Date", "MOM"]
    mom["Date"] = pd.to_datetime(mom["Date"], format="%Y%m")
    mom.set_index("Date", inplace=True)
    mom = mom.astype(float)
    mom /= 100

    # Merge
    data = pd.merge(data, mom, left_index=True, right_index=True)
    data = data[["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]]

    # Select since date
    # since_y = "1963"
    # Or all data
    if not since_y:
        since_y = str(data.index.min().year)
    data = data[since_y:]
    return data

# Sets to first of month
def set_first(df, inplace=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df.loc[:,"Date"] = [f"{y}-{m}" for y, m in zip(df.index.year.to_list(), df.index.month.to_list())]
        df = df.reset_index(drop=True)
        df.loc[:,"Date"] = pd.to_datetime(df.loc[:,"Date"])
        df = df.set_index("Date", drop=True)
        if not inplace:
            return df

if __name__ == '__main__':
    # import os
    # print(os.getcwd())
    print("Testing Web Download.")
    try:
        print(fama_french(from_web=True).head())
    except:
        print("Failed")

    print("Testing Local Download.")
    try:
        print(fama_french(from_web=False).head())
    except:
        print("Failed")