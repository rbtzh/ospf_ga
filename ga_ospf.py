import random
import networkx as nx
import matplotlib.pyplot as plt

# Constants for constraints
Dmax = 100  # Maximum allowed delay
Bmin = 500  # Minimum required bandwidth
Lmax = 0.1  # Maximum allowed loss rate

# Network topology with link attributes
def create_network():
    G = nx.Graph()
    G.add_edge(1, 2, delay=2, loss_rate=0.01, cost=100, bandwidth=1000)
    G.add_edge(1, 3, delay=1, loss_rate=0.02, cost=150, bandwidth=800)
    G.add_edge(1, 4, delay=2, loss_rate=0.025, cost=180, bandwidth=750)
    G.add_edge(2, 3, delay=5, loss_rate=0.015, cost=120, bandwidth=90)
    G.add_edge(2, 4, delay=2, loss_rate=0.03, cost=200, bandwidth=700)
    G.add_edge(3, 5, delay=1, loss_rate=0.035, cost=250, bandwidth=600)
    G.add_edge(4, 6, delay=1, loss_rate=0.04, cost=300, bandwidth=500)
    return G

# Traffic matrix (simplified)
traffic_matrix = {
    (5, 6):100
}

# Calculate path parameters
def calculate_path_parameters(G, path):
    delay = sum(G[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
    bandwidth = min(G[path[i]][path[i+1]]['bandwidth'] for i in range(len(path)-1))
    cost = sum(G[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
    loss_rate = 1 - prod(1 - G[path[i]][path[i+1]]['loss_rate'] for i in range(len(path)-1))
    return delay, bandwidth, cost, loss_rate

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

# Calculate composite weight based on multiple factors
def calculate_composite_weight(delay, bandwidth, cost, loss_rate, weights):
    delay_weight, bandwidth_weight, cost_weight, loss_weight = weights
    
    # Normalize bandwidth (higher bandwidth is better)
    normalized_bandwidth = 1000 / bandwidth  # Assuming 1000 is the max bandwidth
    
    composite_weight = (
        delay_weight * delay +
        bandwidth_weight * normalized_bandwidth +
        cost_weight * cost +
        loss_weight * loss_rate * 1000  # Scale up loss_rate for better balance
    )
    return max(1, int(composite_weight))  # Ensure weight is at least 1

# Fitness function
def calculate_fitness(G, chromosome):
    # Apply link weights from chromosome
    for i, (u, v) in enumerate(G.edges()):
        weights = chromosome[i*4:(i+1)*4]  # Each link has 4 weight factors
        G[u][v]['weight'] = calculate_composite_weight(
            G[u][v]['delay'], G[u][v]['bandwidth'], G[u][v]['cost'], G[u][v]['loss_rate'], weights
        )

    total_cost = 0
    constraint_violations = 0
    for src, dst in traffic_matrix:
        try:
            path = nx.shortest_path(G, src, dst, weight='weight')
            delay, bandwidth, cost, loss_rate = calculate_path_parameters(G, path)
            
            # Check constraints
            if delay > Dmax or bandwidth < Bmin or loss_rate > Lmax:
                constraint_violations += 1
            
            total_cost += cost
        except nx.NetworkXNoPath:
            constraint_violations += 1

    if constraint_violations > 0:
        return 1 / (constraint_violations + 1)  # Penalize solutions that violate constraints
    else:
        return 1 / (total_cost + 1)  # Maximize fitness for lower cost

# Create initial population
def create_population(pop_size, chromosome_length):
    return [random.choices(range(1, 101), k=chromosome_length) for _ in range(pop_size)]

# Selection
def select_parents(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutate(chromosome, mutation_rate):
    return [gene if random.random() > mutation_rate else random.randint(1, 100) for gene in chromosome]

# Genetic Algorithm
def genetic_algorithm(G, pop_size, generations, mutation_rate):
    chromosome_length = len(G.edges()) * 4  # 4 weights per link
    population = create_population(pop_size, chromosome_length)
    for generation in range(generations):
        fitnesses = [calculate_fitness(G, chrom) for chrom in population]
        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        new_population = []
        while len(new_population) < pop_size:
            parents = select_parents(population, fitnesses)
            offspring = crossover(*parents)
            new_population.extend([mutate(child, mutation_rate) for child in offspring])

        population = new_population[:pop_size]

    best_chromosome = max(population, key=lambda chrom: calculate_fitness(G, chrom))
    return best_chromosome

def create_network_topology_plot(G, paths):
    positions = {
        1: (0, 2),
        2: (1, 2),
        3: (0, 1),
        4: (1, 1),
        5: (0, 0),
        6: (1, 0)
    }
    
    fig = plt.figure(figsize=(8, 8))
    
    for idx, ((src, dst), path) in enumerate(zip(traffic_matrix.keys(), paths), 1):
        ax = fig.add_subplot(2, 2, idx)
        
        nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(G, positions, ax=ax, node_size=300, node_color='skyblue')
        nx.draw_networkx_labels(G, positions, ax=ax, font_size=8, font_weight="bold")
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, ax=ax, font_size=6)
        
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, positions, edgelist=path_edges, ax=ax, edge_color='r', width=2)
        
        ax.set_title(f"Path from {src} to {dst}", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("Network Topology with Optimized Paths", fontsize=16)
    plt.tight_layout()
    return fig
# Main execution
if __name__ == "__main__":
    G = create_network()
    pop_size = 50
    generations = 100
    mutation_rate = 0.1

    best_weights = genetic_algorithm(G, pop_size, generations, mutation_rate)
    
    print("\\nBest Link Weights:")
    for i, (u, v) in enumerate(G.edges()):
        weights = best_weights[i*4:(i+1)*4]
        composite_weight = calculate_composite_weight(
            G[u][v]['delay'], G[u][v]['bandwidth'], G[u][v]['cost'], G[u][v]['loss_rate'], weights
        )
        G[u][v]['weight'] = composite_weight
        print(f"Link {u}-{v}: Composite Weight = {composite_weight}")
        print(f"  Delay Weight: {weights[0]}, Bandwidth Weight: {weights[1]}, Cost Weight: {weights[2]}, Loss Rate Weight: {weights[3]}")

    print(f"\\nFinal Fitness: {calculate_fitness(G, best_weights)}")

    print("\\nPath Parameters for Traffic Matrix:")
    paths = []
    for src, dst in traffic_matrix:
        path = nx.shortest_path(G, src, dst, weight='weight', method='dijkstra')
        paths.append(path)
        delay, bandwidth, cost, loss_rate = calculate_path_parameters(G, path)
        print(f"Path {src} to {dst}: {path}")
        print(f"  Delay = {delay}, Bandwidth = {bandwidth}, Cost = {cost}, Loss Rate = {loss_rate:.4f}")
        print(f"  Constraints met: Delay {'✓' if delay <= Dmax else '✗'}, "
              f"Bandwidth {'✓' if bandwidth >= Bmin else '✗'}, "
              f"Loss Rate {'✓' if loss_rate <= Lmax else '✗'}")

    # 创建网络拓扑图
    fig = create_network_topology_plot(G, paths)
    
    # 显示图形
    plt.show()