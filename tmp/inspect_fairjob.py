import pandas as pd
try:
    df = pd.read_csv("hf://datasets/criteo/FairJob/fairjob.csv.gz", nrows=5)
    print("COLUMNS_START")
    for col in df.columns:
        print(col)
    print("COLUMNS_END")
    print("Head:\n", df.head())
except Exception as e:
    print("Error:", e)
