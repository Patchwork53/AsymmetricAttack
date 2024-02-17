Generate blip_results.csv with columns "input_text", "output_text" + others.<br> Add api_key inside deta1_perplexity_difference.py. 

```
python delta1_perplexity_difference.py --csv_file blip_results.csv
python delta2_baseline_difference.py --csv_file blip_results.csv
```

Two columns perp_diff and baseline_diff will be added to the csv.

