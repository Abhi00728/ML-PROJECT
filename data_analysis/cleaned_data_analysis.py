import pandas as pd

df = pd.read_parquet("../data/lending_clean.parquet")

miss_by_year = df.groupby("issue_year").apply(
    lambda g: (g.isna().mean() * 100).round(2), include_groups=False
).T
miss_by_year["overall_pct"] = (df.isna().mean() * 100).round(2)
miss_by_year["diff_2015_vs_2016"] = (miss_by_year[2015] - miss_by_year[2016]).abs().round(2)

# Show all columns with any missing values, highlighting time-gated 2015 vs 2016 shifts
print(miss_by_year[miss_by_year["overall_pct"] > 0].sort_values("diff_2015_vs_2016", ascending=False).to_string())