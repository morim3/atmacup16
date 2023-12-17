import os
from typing import Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm
import polars as pl


def find_closest_nodes_bfs(G, n=10):
    """
    Find the closest 'n' nodes for each node in the graph 'G' using BFS.
    Returns a dictionary where keys are nodes and values are lists of closest 'n' nodes.
    """
    closest_nodes = {}
    for node in tqdm(list(G.nodes())):
        # Perform BFS from the node
        bfs_tree = nx.bfs_tree(G, node, )

        closest = []
        for i, b in enumerate(bfs_tree):
            if i == n:
                break
            if i > 0:
                closest.append(b)

        closest_nodes[node] = closest

    return closest_nodes

def co_count_yad(
    train_log: pl.DataFrame,
    test_log: Optional[pl.DataFrame] = None,
) -> pd.DataFrame:
    # test_logが引数に与えられていたらtrain_logと結合する
    if test_log is not None:
        log = pl.concat(
            [train_log[["session_id", "yad_no"]], test_log[["session_id", "yad_no"]]]
        )
    else:
        log = train_log[["session_id", "yad_no"]].copy()

    log = log.join(log, on="session_id")
    # yad_noが同じものは除外する
    log = log.filter(pl.col("yad_no") != pl.col("yad_no_right"))

    log = log.group_by(["yad_no", "yad_no_right"]).count()

    log = log.rename(
        {
            "yad_no_right":"candidate_yad_no",
            "count":"co_visit_count",
        }
    )[["yad_no", "candidate_yad_no", "co_visit_count"]]

    return log


if __name__ == "__main__":

    train_log = pl.read_parquet("data/train_log.parquet")
    test_log = pl.read_parquet("data/test_log.parquet")

    cocount_log = co_count_yad(train_log, test_log)

    G = nx.Graph()
    for u, v, weight in cocount_log[["yad_no", "candidate_yad_no", "co_visit_count"]].to_numpy():
        G.add_edge(u, v, weight=1)


    for top in [20, 30]:
        candidates_dict = find_closest_nodes_bfs(G, n=top)

        yad_no_list = []
        candidates_list = []
        for key, values in candidates_dict.items():
            yad_no_list.extend([key] * len(values))
            candidates_list.extend(values)

        candidates_df = pl.DataFrame({"yad_no": yad_no_list, "candidate_yad_no": candidates_list})


        candidates_df.write_parquet(f"data/candidates/train_network_top{top}_candidates.parquet", )


