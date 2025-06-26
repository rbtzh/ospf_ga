import matplotlib.pyplot as plt

# 初始化图拓扑
network = {
    0: {1: 100, 2: 150, 3: 180},
    1: {0: 100, 2: 120, 3: 200},
    2: {0: 150, 1: 120, 4: 250},
    3: {0: 180, 1: 200, 5: 300},
    4: {2: 250},
    5: {3: 300}
}

# 初始化路由表，专门针对节点4
def initialize_routing_table_for_node_4(num_nodes):
    routing_table = {4: {0: float('inf'), 1: float('inf'), 2: float('inf'), 3: float('inf'), 4: 0, 5: float('inf')}}
    return routing_table

# 更新路由表，只计算从节点4到其他节点的最短路径
def update_routing_table_for_node_4(routing_table, network):
    updated = True
    while updated:
        updated = False
        for node in network:
            if node == 4:
                continue
            for neighbor in network[node]:
                new_distance = routing_table[4][node] + network[node][neighbor]
                if new_distance < routing_table[4][neighbor]:
                    routing_table[4][neighbor] = new_distance
                    updated = True
    return routing_table

# 打印节点4的路由表
def print_routing_table_for_node_4(routing_table):
    print("路由器 4 的路由表:")
    for dest in routing_table[4]:
        print("  到路由器 {} 的最短距离为: {}".format(dest, routing_table[4][dest]))
    print()

# 绘制从节点4到目标节点的拓扑图
def draw_path_topology(network, target_node, routing_table, ax):
    positions = {
        0: (0, 2), 1: (1, 2), 2: (0, 1),
        3: (1, 1), 4: (0, 0), 5: (1, 0)
    }

    # 计算最短路径
    path = []
    if target_node in routing_table[4]:
        current_node = target_node
        while current_node != 4:
            path.append(current_node)
            for neighbor in network:
                if neighbor in network[current_node] and routing_table[4][neighbor] + network[neighbor][current_node] == routing_table[4][current_node]:
                    current_node = neighbor
                    break
        path.append(4)
        path = path[::-1]

    # 绘制所有节点
    for node, pos in positions.items():
        ax.scatter(*pos, s=1000, color='lightgray', edgecolor='gray', linewidth=2)
        ax.text(pos[0], pos[1], '{}'.format(node), fontsize=15, ha='center', va='center')

    # 绘制边和cost值
    drawn_edges = set()
    for node in network:
        for neighbor in network[node]:
            if (node, neighbor) not in drawn_edges and (neighbor, node) not in drawn_edges:
                drawn_edges.add((node, neighbor))
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]
                
                on_path = node in path and neighbor in path and abs(path.index(node) - path.index(neighbor)) == 1
                
                color = 'blue' if on_path else 'lightgray'
                linewidth = 2 if on_path else 1
                zorder = 3 if on_path else 1
                
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=zorder)
                
                text_color = 'red' if on_path else 'gray'
                fontsize = 12 if on_path else 10
                fontweight = 'bold' if on_path else 'normal'
                
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                offset = 0.05
                dx = x2 - x1
                dy = y2 - y1
                perp_x = -dy * offset
                perp_y = dx * offset
                
                ax.text(mid_x + perp_x, mid_y + perp_y, str(network[node][neighbor]),
                        fontsize=fontsize, color=text_color, fontweight=fontweight,
                        ha='center', va='center', zorder=4)

    # 高亮起点和终点，并添加src和dst标注
    src_pos = positions[4]
    dst_pos = positions[target_node]
    
    ax.scatter(*src_pos, s=1000, color='lightgreen', edgecolor='green', linewidth=2, zorder=5)
    ax.scatter(*dst_pos, s=1000, color='lightyellow', edgecolor='orange', linewidth=2, zorder=5)
    
    # 添加src标注
    ax.text(src_pos[0], src_pos[1] + 0.15, 'src', fontsize=12, ha='center', va='bottom', color='green', fontweight='bold')
    
    # 添加dst标注
    ax.text(dst_pos[0], dst_pos[1] + 0.15, 'dst', fontsize=12, ha='center', va='bottom', color='orange', fontweight='bold')

    # 设置子图的显示属性
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Path 4 -> {}'.format(target_node))


# 修改distance_vector_routing_simulation函数
def distance_vector_routing_simulation(network):
    num_nodes = len(network)
    routing_table = initialize_routing_table_for_node_4(num_nodes)
    
    # 初始化邻居之间的距离
    for neighbor in network[4]:
        routing_table[4][neighbor] = network[4][neighbor]
    
    # 运行路由更新
    routing_table = update_routing_table_for_node_4(routing_table, network)
    
    # 输出节点4的路由表
    print_routing_table_for_node_4(routing_table)
    
    # 创建一个大的figure来容纳所有子图，并指定figure的尺寸
    fig = plt.figure(figsize=(15, 10))  # 设置figure的宽度为15英寸，高度为10英寸
    
    # 分别绘制从4到其他节点的路径拓扑图
    for i, target_node in enumerate(range(num_nodes)):
        if target_node != 4:
            ax = fig.add_subplot(2, 3, i+1)  # 2行3列的布局
            draw_path_topology(network, target_node, routing_table, ax)

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

# 运行模拟
distance_vector_routing_simulation(network)