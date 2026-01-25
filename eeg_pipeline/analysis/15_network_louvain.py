# eeg_pipeline/analysis/15_network_louvain.py
"""
Step 15: Network Analysis with Louvain Community Detection
Creates correlation matrix of nodal PAC changes and detects communities.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

FEATURES_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR = pipeline_dir / "outputs" / "network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Node names (must match PAC output)
NODE_NAMES = ['LF', 'CF', 'RF', 'LC', 'C', 'RC', 'LP', 'CP', 'RP']

try:
    import community as community_louvain
    import networkx as nx
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("Warning: python-louvain not installed. Install with: pip install python-louvain")


def main():
    pac_file = FEATURES_DIR / "pac_nodal_features.csv"
    
    if not pac_file.exists():
        print(f"PAC features not found: {pac_file}")
        print("Run Step 13 first.")
        return
    
    df = pd.read_csv(pac_file)
    print(f"Loaded PAC data for {len(df)} subjects")
    
    # Extract PAC columns
    pac_cols = [f'pac_{node}' for node in NODE_NAMES]
    available_cols = [c for c in pac_cols if c in df.columns]
    
    if len(available_cols) < 3:
        print("Not enough PAC nodes for network analysis")
        return
    
    pac_data = df[available_cols].values
    
    # Remove subjects with NaN
    valid_mask = ~np.any(np.isnan(pac_data), axis=1)
    pac_data = pac_data[valid_mask]
    print(f"Valid subjects for network analysis: {pac_data.shape[0]}")
    
    if pac_data.shape[0] < 3:
        print("Not enough valid subjects for correlation matrix")
        return
    
    # Compute 9x9 correlation matrix (which nodes change together)
    corr_matrix = np.corrcoef(pac_data.T)
    
    # Save correlation matrix
    corr_df = pd.DataFrame(corr_matrix, index=NODE_NAMES[:len(available_cols)], 
                           columns=NODE_NAMES[:len(available_cols)])
    corr_df.to_csv(OUTPUT_DIR / "pac_correlation_matrix.csv")
    print(f"Saved correlation matrix to {OUTPUT_DIR / 'pac_correlation_matrix.csv'}")
    
    if not LOUVAIN_AVAILABLE:
        print("Louvain not available - skipping community detection")
        return
    
    # Create network graph (threshold correlations)
    threshold = 0.3  # Only keep correlations above threshold
    G = nx.Graph()
    
    for i, node_i in enumerate(NODE_NAMES[:len(available_cols)]):
        G.add_node(node_i)
        for j, node_j in enumerate(NODE_NAMES[:len(available_cols)]):
            if i < j and abs(corr_matrix[i, j]) > threshold:
                G.add_edge(node_i, node_j, weight=abs(corr_matrix[i, j]))
    
    # Louvain community detection
    if len(G.edges()) > 0:
        partition = community_louvain.best_partition(G)
        
        # Save community assignments
        community_df = pd.DataFrame([
            {'node': node, 'community': comm}
            for node, comm in partition.items()
        ])
        community_df.to_csv(OUTPUT_DIR / "community_assignments.csv", index=False)
        print(f"\nCommunity assignments:")
        for comm_id in set(partition.values()):
            members = [n for n, c in partition.items() if c == comm_id]
            print(f"  Community {comm_id}: {members}")
        
        # Compute within-community strength
        communities = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        
        strengths = {}
        for comm_id, members in communities.items():
            if len(members) > 1:
                member_idx = [NODE_NAMES.index(m) for m in members if m in NODE_NAMES]
                within_corrs = [corr_matrix[i, j] for i in member_idx for j in member_idx if i < j]
                strengths[comm_id] = np.mean(within_corrs) if within_corrs else 0
        
        print(f"\nWithin-community strength: {strengths}")
    else:
        print("No edges above threshold - network is sparse")


if __name__ == "__main__":
    main()
