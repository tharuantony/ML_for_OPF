import pickle
import json

# Load pickle file
with open('/Users/tharuantonymelath/Documents/Master_Thesis/results/dec-results1.pkl', 'rb') as file:
    results = pickle.load(file)

print("=== Summary of Pickle File ===\n")

# Separate settings / metadata and node-level entries
for key, value in results.items():
    if isinstance(value, dict):
        # Probably a node-level entry
        print(f"Node ID: {key}")
        for category, metrics in value.items():
            if isinstance(metrics, dict):
                print(f"  {category}:")
                for metric, stats in metrics.items():
                    if isinstance(stats, dict) and 'avg' in stats and 'n' in stats:
                        print(f"    {metric}: avg = {stats['avg']}, n = {stats['n']}")
                    else:
                        print(f"    {metric}: {stats} (unexpected format)")
            else:
                print(f"  {category}: {metrics} (unexpected format)")
        print("\n" + "-"*50 + "\n")
    else:
        # Global setting or metadata
        print(f"Global Key: {key}")
        try:
            print(json.dumps(value, indent=4))  # Pretty print if it's JSON-compatible
        except:
            print(f"  {value}")
        print("\n" + "="*50 + "\n")
