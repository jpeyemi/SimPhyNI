# scripts/reformat_tree.py
import sys
from ete3 import Tree

def process_tree(input_tree, output_tree):
    try:
        tree = Tree(input_tree)
    except:
        tree = Tree(input_tree, 1)

    tree.name = 'root'
    n_clamped = 0
    for idx, node in enumerate(tree.iter_descendants("levelorder")):
        if not node.is_leaf() and not node.name:
            node.name = f"internal_{idx}"
            print(f"Unnamed node renamed to {node.name}")
        if node.dist < 0:
            node.dist = 0.0
            n_clamped += 1

    if n_clamped:
        print(f"Clamped {n_clamped} negative branch lengths to 0.0")

    tree.write(format=1, outfile=output_tree)

if __name__ == "__main__":
    process_tree(sys.argv[1], sys.argv[2])
    