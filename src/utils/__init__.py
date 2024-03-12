import pandas as pd

def history2df(history : dict) -> pd.DataFrame:
    
    dfs = []

    for key in history.keys():
        df = pd.DataFrame(history[key])
        df["split"] = [key for _ in range(len(df))]
        dfs.append(df)

    return pd.concat(dfs)