import pandas as pd
f='d:/astra/tradebot/data/generated/wf_portfolio_explore_runs/20260316_025259/explore_runs.csv'
df=pd.read_csv(f)
df.drop_duplicates(subset=['label_id', 'model_id', 'backtest_id', 'stage'], keep='last', inplace=True)
df.to_csv(f, index=False)
print('Deduplicated')
