import pandas as pd

# Load and display statistics
df = pd.read_csv('sav2.csv')
df.columns = df.columns.str.strip()

print("\n" + "="*60)
print("PREDICTION DATA STATISTICS (sav2.csv)")
print("="*60)

print(f"\nTotal samples: {len(df)}")
print(f"Unique frames: {df['frame'].nunique()}")
print(f"Unique students (track_id): {df['track_id'].nunique()}")

print("\n" + "-"*60)
print("ENGAGEMENT LEVEL DISTRIBUTION:")
print("-"*60)
dist = df['engagement_level'].value_counts()
for level, count in dist.items():
    pct = (count / len(df)) * 100
    print(f"  {level:10s}: {count:5d} ({pct:5.2f}%)")

print("\n" + "="*60)
