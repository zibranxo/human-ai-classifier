import json

def generate_dot(tree):
    dot = ["digraph Tree {", 'node [shape=box, style="filled"];']
    node_id = 0

    def traverse(node, parent=None, edge_label=""):
        nonlocal node_id
        current_id = f"node{node_id}"
        node_id += 1

        if node.get("leaf", False):
            label = f'label: {node["label"]}'
        else:
            label = node["feature"]

        dot.append(f'{current_id} [label="{label}"];')

        if parent:
            dot.append(f'{parent} -> {current_id} [label="{edge_label}"];')

        if not node.get("leaf", False):
            for branch, child in node["branches"].items():
                traverse(child, current_id, branch)

    traverse(tree)
    dot.append("}")
    return "\n".join(dot)


# Load your file
with open("id3_tree.json") as f:
    tree = json.load(f)

dot_code = generate_dot(tree)

with open("tree.dot", "w") as f:
    f.write(dot_code)