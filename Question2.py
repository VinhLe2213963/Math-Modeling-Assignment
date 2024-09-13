import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque


INF = 1e9

# Edge: Set of edge on our network
class Edge:
    def __init__(self, from_, to, capacity, cost):
        self.from_ = from_
        self.to = to
        self.capacity = capacity
        self.cost = cost

# SPFA Algorithm (Improved Bellman-Ford) to get the shortest path
def shortest_paths(n, v0, adj, cost, capacity):
    # d: distance from source to other nodes
    d = [INF]*n
    d[v0] = 0
    
    # inqueue: To keep track of which nodes are currently in the queue
    inq = [False]*n
    
    # q: queue for shortest path algorithm
    q = deque([v0])
    
    # p: Keep track of the path by store the previous node of the current node
    p = [-1]*n
    
    # count: Counter to check for Negative Cycle 
    count = [0]*n
    
    while q:
        u = q.popleft()
        inq[u] = False
        for v in adj[u]:
            if capacity[u][v] > 0 and d[v] > d[u] + cost[u][v]:
                d[v] = d[u] + cost[u][v]
                p[v] = u
                if not inq[v]:
                    q.append(v)
                    inq[v] = True
                    count[v] += 1
                    if count[v] > n:
                        return None  # Negative cycle 

    return d, p

def min_cost_flow(N, edges, K, s, t):
    print("Total Flow: ", K)
    
    # Create adjacency list
    adj = [[] for _ in range(N)]
    cost = [[0]*N for _ in range(N)]
    capacity = [[0]*N for _ in range(N)]
    
    # Create residual network
    for e in edges:
        adj[e.from_].append(e.to)
        adj[e.to].append(e.from_)
        cost[e.from_][e.to] = e.cost
        cost[e.to][e.from_] = -e.cost
        capacity[e.from_][e.to] = e.capacity

    # Initialize the result variables
    flow = 0
    cost_ = 0
    
    
    while flow < K:
        result = shortest_paths(N, s, adj, cost, capacity)
        
        # Case 1: Negative Cycle
        if result is None:
            raise ValueError("Negative cycle detected")
        d, p = result
        
        # Case 2: There is no shortest path from source to sink
        if d[t] == INF:
            break;

        # Case 3: Shortest path exists
        f = K - flow
        cur = t
        path = []
        while cur != s:
            f = min(f, capacity[p[cur]][cur])
            path.append(cur)
            cur = p[cur]
        path.append(s)
        path = path[::-1]

        # Print out the shortest path
        print('Path:', ' -> '.join(map(str, path)))
        print('Flow sent on this path: ', f)
        print('Cost per flow on this path: ', d[t])
        print("________________________________________________________________")

        # Update the residual network
        flow += f
        cost_ += f * d[t]
        cur = t
        while cur != s:
            capacity[p[cur]][cur] -= f
            capacity[cur][p[cur]] += f
            cur = p[cur]

    # Return the answer
    if flow < K:
        return -1
    else:
        return cost_



# Create grid network 5x10 (50 nodes)
G = nx.grid_2d_graph(5, 10)

# Add cost and capacity for each edge
for (u, v) in G.edges():
    G.edges[u, v]['cost'] = random.randint(1, 10)  # Cost from 1 to 10
    G.edges[u, v]['capacity'] = random.randint(200, 700)  # Capacity from 200 to 700

# Change the node number to 0-49
mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

# Fix the position of the grid network on 2-D plane
pos = {i: (i % 10, 4 - i // 10) for i in range(50)}

# Set the color the each node
source_node = 0  # Source is node 0
target_node = 49  # Sink is node 49
node_colors = ['salmon' if node == source_node else 'lightgreen' if node == target_node else 'skyblue' for node in G.nodes()]
# Draw the network
nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=8)

# Create label for each edge with the format "cost | capacity"
edge_labels = {(u, v): f"{G.edges[u, v]['cost']} | {G.edges[u, v]['capacity']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', rotate=False)


# Create the graph based on the grid network
edges = []
for u, v, attr in G.edges(data=True):
    edges.append(Edge(u, v, attr['capacity'], attr['cost']))
    edges.append(Edge(v, u, 0, -attr['cost']))  # Reverse edge in residual network

# Determine the source and sink
s = 0  # Source is node 0
t = 49  # Sink is node 49

# The amount of flow to be sent from source to sink
K = 893  # For example: Flow is 893
N = len(G.nodes())

# Call the min-cost flow function
cost = min_cost_flow(N, edges, K, s, t)
print("Chi phí tối thiểu để chuyển flow là:", cost)


# Plot the network
plt.show()