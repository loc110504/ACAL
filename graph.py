# file: demo_draw_graph.py

import networkx as nx
import matplotlib.pyplot as plt

# Khởi tạo directed graph
G = nx.DiGraph()

# Các node (tên tương ứng với workflow của bạn)
nodes = [
    "rag_retrieval",
    "team_selections_and_define_criteria",
    "argument_generation",
    "human_review",
    "argument_validation",
    "final_answer_generation",
    "END",
]

# Thêm nodes
G.add_nodes_from(nodes)

# Thêm các cạnh (edges)
edges = [
    ("rag_retrieval", "team_selections_and_define_criteria"),
    ("team_selections_and_define_criteria", "argument_generation"),
    ("argument_generation", "human_review"),
    # conditional edges from human_review
    ("human_review", "human_review"),            # nếu review chưa ok, quay lại human_review — bạn có thể loại bỏ nếu không muốn loop
    ("human_review", "argument_validation"),     # nếu review ok -> validate
    ("argument_validation", "final_answer_generation"),  # nếu validate ok -> final answer
    ("final_answer_generation", "END"),
]

G.add_edges_from(edges)

# Vẽ graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)  # bố cục tự động
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", arrowsize=20, arrowstyle='-|>')
plt.title("Legal Decision-Making / Argumentation Workflow (demo)")
plt.axis("off")
plt.savefig("workflow_graph.png", bbox_inches="tight", dpi=200)
plt.close()
