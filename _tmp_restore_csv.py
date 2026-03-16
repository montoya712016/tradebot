import pandas as pd
from io import StringIO
import re
import os

repo_root = "d:/astra/tradebot"
log_path = r"C:\Users\NovoLucas\.gemini\antigravity\brain\b171e269-cc3f-43ab-a4ad-0eef31ed7d61\.system_generated\logs\overview.txt"
csv_path = "d:/astra/tradebot/data/generated/wf_portfolio_explore_runs/20260316_025259/explore_runs.csv"

# The issue is we accidentally dropped `keep=last` on duplicated label_ids.
# We need `keep=first` to restore the good runs.
# BUT we ran that python script in-place with `.to_csv(f, index=False)`.
# So the original file was overwritten. Let's see if the old data is in the overview log.

def run():
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

    # The file contents were logged via `view_file` tool call. Let's find "1: label_id,model_id..."
    # and extract it.
    
    start_tag = "1: label_id,model_id,backtest_id,train_id,stage,status,start_utc,end_utc,duration_sec,seed,"
    end_tag = "The above content shows the entire, complete file contents of the requested file."
    
    if start_tag in log_content:
        idx1 = log_content.find(start_tag)
        idx2 = log_content.find(end_tag, idx1)
        
        csv_block = log_content[idx1:idx2]
        
        # Clean the block
        lines = []
        for line in csv_block.split('\n'):
            line = line.strip()
            if not line: continue
            
            # Remove "XX: " from start
            match = re.match(r"^\d+:\s(.*)$", line)
            if match:
                lines.append(match.group(1))
                
        # Now we have the original lines (which had the duplicates). We'll save them,
        # and then deduplicate with `keep='first'`.
        
        df = pd.read_csv(StringIO("\n".join(lines)))
        
        # We also need to add back the latest runs that happened AFTER I read the file
        # those are currently safe in actual explore_runs.csv on disk, let's load it too.
        df_current = pd.read_csv(csv_path)
        
        # Merge them
        df_merged = pd.concat([df, df_current], ignore_index=True)
        
        # Drop duplicates based on the identifiers. 
        # Crucially, we use `keep='first'` so we keep `df`'s versions (the old good ones)
        # over `df_current`'s versions (the new bad ones with the same IDs)
        # However, for newly generated labels like label_004, they will be kept.
        
        df_merged.drop_duplicates(subset=['label_id', 'model_id', 'backtest_id', 'stage'], keep='first', inplace=True)
        
        # Sort by start_utc
        df_merged['start_utc_dt'] = pd.to_datetime(df_merged['start_utc'])
        df_merged.sort_values(by='start_utc_dt', inplace=True)
        df_merged.drop(columns=['start_utc_dt'], inplace=True)
        
        df_merged.to_csv(csv_path, index=False)
        print("Restored original good runs from logs and rebuilt explore_runs.csv!")
        
    else:
        print("Could not find the original CSV block in the logs.")

if __name__ == "__main__":
    run()
