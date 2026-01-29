import json
import numpy as np
import collections

def decode_cluster_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Invert the dictionary: Cluster ID -> List of Tactic Strings
    clusters = collections.defaultdict(list)
    for tactic_str, cluster_id in data['by_tactic_string'].items():
        clusters[cluster_id].append(tactic_str)
        
    print(f"{'='*60}")
    print(f"BEHAVIOR CLUSTER DECODER (Total Clusters: {len(clusters)})")
    print(f"{'='*60}\n")
    
    # Feature Labels corresponding to indices 0-5
    feature_names = [
        "Box In Front (Contact)",  # Index 0
        "Box In Front Row",        # Index 1
        "Box On Sides",            # Index 2
        "Box In Back",             # Index 3
        "Wall Nearby",             # Index 4
        "Action: Turn"             # Index 5 (0=Move, 1=Turn)
    ]
    
    for c_id in sorted(clusters.keys()):
        tactics = clusters[c_id]
        count = len(tactics)
        
        # Calculate feature frequencies
        # Tactic format: "BFSKW_A" (6 chars used for features)
        # We need to parse the binary logic again
        feature_counts = np.zeros(6)
        
        for t in tactics:
            code, action = t.split('_')
            # 0: B vs C
            if code[0] == 'B': feature_counts[0] += 1
            # 1: F vs -
            if code[1] == 'F': feature_counts[1] += 1
            # 2: S vs -
            if code[2] == 'S': feature_counts[2] += 1
            # 3: K vs -
            if code[3] == 'K': feature_counts[3] += 1
            # 4: W vs O
            if code[4] == 'W': feature_counts[4] += 1
            # 5: Action T vs F
            if action == 'T': feature_counts[5] += 1
            
        print(f"CLUSTER {c_id} ({count} tactics)")
        print(f"{'-'*30}")
        
        # Determine Dominant Features (>80% or <20%)
        dominant_traits = []
        for i, freq in enumerate(feature_counts):
            pct = (freq / count) * 100
            if pct > 85:
                dominant_traits.append(f"ALWAYS {feature_names[i]}")
            elif pct < 15:
                dominant_traits.append(f"NEVER {feature_names[i]}")
        
        if not dominant_traits:
            print("  (Mixed/Transitional Cluster)")
        else:
            for trait in dominant_traits:
                print(f"  • {trait}")
        
        # Sample Tactic
        print(f"  Sample: {tactics[0]}")
        print("\n")

if __name__ == "__main__":
    # Assuming the file is in the same directory
    try:
        decode_cluster_file('behavior_clusters.json')
    except FileNotFoundError:
        print("Please ensure 'behavior_clusters.json' is in the directory.")