import pandas as pd
df = pd.read_csv('d:/astra/tradebot/data/generated/fair_wf_explore_v6/robustness_report/walkforward_oos_segments_reuse.csv')
for i, row in df.iterrows():
    print(f"source: {row['source_step']}d -> test: {row['test_start_step']}d to {row['target_step']}d | Window: {row['window_info']}")
