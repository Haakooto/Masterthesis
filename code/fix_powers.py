import pandas as pd
from glob import glob

def fix(filename):
    df = pd.read_csv(filename)
    df.loc[(df["activate"] == "rexp") | (df["activate"] == "exp"), "power"] = 0
    df.to_csv(filename, index=False)

def main():
    for file in glob("csvs/*.csv"):
        fix(file)

if __name__ == "__main__":
    main()