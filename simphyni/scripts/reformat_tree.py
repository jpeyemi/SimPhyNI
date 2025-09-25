# scripts/reformat_tree.py
import sys
from ete3 import Tree

def process_tree(input_tree, output_tree):
    try:
        tree = Tree(input_tree)
    except:
        tree = Tree(input_tree, 1)

    tree.name = 'root'
    for idx, node in enumerate(tree.iter_descendants("levelorder")):
        if not node.is_leaf() and not node.name:
            node.name = f"internal_{idx}"
            print(f"Unnamed node renamed to {node.name}")

    tree.write(format=1, outfile=output_tree)

if __name__ == "__main__":
    process_tree(sys.argv[1], sys.argv[2])
    