import pandas as pd


def make_window(df,window=5):

    rows=[]

    for company in df['company'].unique():

        temp=df[df.company==company]
        temp=temp.sort_values('year')

        for i in range(len(temp)-window):

            chunk=temp.iloc[i:i+window]
            target=temp.iloc[i+window]

            row={}

            for idx,r in enumerate(chunk.itertuples()):

                row[f'revenue_{idx}']=r.revenue
                row[f'op_{idx}']=r.operating_profit
                row[f'net_{idx}']=r.net_income
                row[f'debt_{idx}']=r.debt
                row[f'equity_{idx}']=r.equity
                row[f'cash_{idx}']=r.cash

            row['next_operating_profit']=target.operating_profit

            rows.append(row)

    return pd.DataFrame(rows)