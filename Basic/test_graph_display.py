import networkx as nx
import matplotlib.pyplot as plt
#from applications.optimization.Backtracking.data.graph_coloring import problem_in

nodes = [2, -1, 1,-1 ,3,1,-1, 2, -1]
edges_list = [[0,2],[0,4],[0,5],[0,7],[1,2],[1,3],[1,4],[1,5],[1,7],[2,3],[2,4],[2,6],[2,8],[3,4],[3,5],[3,7],[4,6],[4,8],[5,6],[5,8],[6,7],[7,8]]

edges = [tuple(edge) for edge in edges_list]


#found solution
sol = [2, 0, 3]

problem_in = [nodes, edges]


G = nx.Graph()
color_map = []
for node in problem_in[0]:
    if node == 0 :
        color_map.append('gray')
    elif node == 1 :
        color_map.append('blue')
    elif node ==2: 
        color_map.append('green')      
    elif node ==3 :
        color_map.append('red')
    else: 
        color_map.append('white')  
G_nodes = list(range(len(problem_in[0])))
print(G_nodes)
G.add_nodes_from(G_nodes)
G.add_edges_from(problem_in[1])
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()



""" 
G2 = nx.Graph()
color_map2 = []
sol_index = 0
for node in problem_in[0]:
    
    if node == -1 :
        node2 = sol[sol_index]
        if node2 == 0 :
            color_map2.append('gray')
        elif node2 == 1 :
            color_map2.append('blue')
        elif node2 ==2: 
            color_map2.append('green')      
        elif node2 ==3 :
            color_map2.append('red')
        sol_index +=1

    if node == 0 :
        color_map2.append('gray')
    elif node == 1 :
        color_map2.append('blue')
    elif node ==2: 
        color_map2.append('green')      
    elif node ==3 :
        color_map2.append('red')

G_nodes2 = list(range(len(problem_in[0])))
print(G_nodes2)
G2.add_nodes_from(G_nodes2)
G2.add_edges_from(problem_in[1])
nx.draw(G2, node_color=color_map2, with_labels=True)
plt.show()

 """