import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import sys

from networkx.algorithms.cluster import average_clustering
from networkx.classes.graph import Graph

from collections import defaultdict
 
class DijkstraGraph:
    def __init__(self):
        self.printArr = list()

    def minDistance(self,dist,queue):
        minimum = float("Inf")
        min_index = -1
         
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index
 
    def printPath(self, parent, j):
         
        if parent[j] == -1 :
            self.printArr.append(j)
            # print(j)
            return
        self.printPath(parent , parent[j])
        self.printArr.append(j)
        # print(j)
         
    def printSolution(self, dist, parent):
        dijkstraNXG = nx.DiGraph()
        temp_pos = nx.get_node_attributes(DG, 'pos')
        for i in range(verts):
            dijkstraNXG.add_node(i, pos = temp_pos)

        src = starting
        # print("Vertex \t\tDistance from Source\tPath")
        for i in range(0, len(dist)):
            if i != src:
                # print("\n%d --> %d \t\t%f \t\t\t\t\t" % (src, i, dist[i]/10000000)),
                self.printArr = list()
                self.printPath(parent,i)

                # self.printArr.reverse()
                for j in range(1, len(self.printArr)):
                    if not dijkstraNXG.has_edge(self.printArr[j-1], self.printArr[j]):
                        dijkstraNXG.add_edge(self.printArr[j-1], self.printArr[j], weight = digraphMat[ self.printArr[j-1] ][ self.printArr[j] ]/10000000)

        nx.draw(dijkstraNXG, temp_pos, with_labels=True)
        plt.savefig("DijkstraGraph.png")
        labels = nx.get_edge_attributes(dijkstraNXG,'weight')
        nx.draw_networkx_edge_labels(dijkstraNXG,temp_pos,edge_labels=labels)
        plt.savefig("DijkstraGraph_with_Weights.png")
        plt.close()

    def dijkstra(self, graph, src):
 
        row = len(graph)
        col = len(graph[0])

        dist = [float("Inf")] * row

        parent = [-1] * row

        dist[src] = 0
     
        queue = []
        for i in range(row):
            queue.append(i)
             
        while queue:

            u = self.minDistance(dist,queue)
            queue.remove(u)

            for i in range(col):
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u
 
        self.printSolution(dist,parent)

class FloydWarshall():
    def __init__(self, Vertices, graph):
        self.V = Vertices
        self.INF = float('inf')
        self.distance = list(map(lambda i: list(map(lambda j: j, i)), graph))

    def algo(self):
        for k in range(self.V):  
            for i in range(self.V):
                for j in range(self.V):
                    self.distance[i][j] = min(self.distance[i][j], self.distance[i][k] + self.distance[k][j])

        return self.distance

    def MakeGraph(self, png_name):
        plt.imshow(self.distance, interpolation='nearest')
        plt.savefig(png_name)
        plt.close() 

class KruskalGraph():
    def __init__(self, Vertices, Starting_Vert):
        self.V = Vertices
        self.S = Starting_Vert
        self.TotalWeight = 0
        self.INF = float('inf')
        self.graph = [[self.INF for i in range(self.V)] for j in range(self.V)]
        self.parent = [i for i in range(self.V)]

    def getTotalWeight(self):
        return self.TotalWeight

    def find(self, i):
        while self.parent[i] != i:
            i = self.parent[i]
        return i

    def union(self, i, j):
        a = self.find(i)
        b = self.find(j)
        self.parent[a] = b

    def kruskalMST(self):
        mincost = 0 # Cost of min MST
 
        # Initialize sets of disjoint sets
        for i in range(self.V):
            self.parent[i] = i

        temp_pos = nx.get_node_attributes(G, 'pos')
        MST = nx.Graph()
        for i in range(self.V):
            MST.add_node(i, pos = temp_pos[i])
    
        # Include minimum weight edges one by one
        edge_count = 0
        while edge_count < self.V - 1:
            min = self.INF
            a = -1
            b = -1
            for i in range(self.V):
                for j in range(self.V):
                    if self.find(i) != self.find(j) and self.graph[i][j] < min:
                        min = self.graph[i][j]
                        a = i
                        b = j
            self.union(a, b)
            #print('Edge {}:({}, {}) cost:{}'.format(edge_count, a, b, min))
            MST.add_edge(a, b, weight = min/10000000)
            edge_count += 1
            mincost += min/10000000

        self.TotalWeight = mincost
        #print("Minimum cost= {}".format(mincost))

        MSTpos=nx.get_node_attributes(MST,'pos')

        min_x = min_y = float('inf')
        for i in range(MST.number_of_nodes()):
            x, y = MSTpos[i]
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y

        nx.draw(MST, MSTpos, with_labels=True, connectionstyle="arc3,rad=0.1")
        plt.text(min_x, min_y-0.012, 'MST = ' + str(round(self.TotalWeight, 2)), fontsize = 22, horizontalalignment= "left", verticalalignment = "top")
        plt.savefig("KruskalMST.png")
        labels = nx.get_edge_attributes(MST,'weight')
        nx.draw_networkx_edge_labels(MST,temp_pos,edge_labels=labels)
        plt.savefig("KruskalMST_with_Weights.png")
        plt.close()
    

class PrimGraph():
 
    def __init__(self, Vertices, Starting_Vert):
        self.V = Vertices
        self.S = Starting_Vert
        self.TotalWeight = 0
        self.graph = [[0 for i in range(Vertices)] for j in range(Vertices)]

    def getTotalWeight(self):
        return self.TotalWeight
 
    def printMST(self, parent):
        #print("Edge \tWeight")
        # for i in range(1, self.V):
        #     print(parent[i], "-", i, "\t", self.graph[i][ parent[i] ]/10000000)

        temp_pos = nx.get_node_attributes(G, 'pos')
        MST = nx.Graph()
        for i in range(self.V):
            MST.add_node(i, pos = temp_pos[i])

        for i in range(1, self.V):
            MST.add_edge(parent[i], i, weight = self.graph[i][parent[i]]/10000000)
            self.TotalWeight += self.graph[i][parent[i]]/10000000

        MSTpos=nx.get_node_attributes(MST,'pos')

        min_x = min_y = float('inf')
        for i in range(MST.number_of_nodes()):
            x,y = MSTpos[i]
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y

        nx.draw(MST, MSTpos, with_labels=True, connectionstyle="arc3,rad=0.1")
        plt.text(min_x, min_y-0.012, 'MST = ' + str(round(self.TotalWeight, 2)), fontsize = 22, horizontalalignment= "left", verticalalignment = "top")
        plt.savefig("PrimMST.png")
        labels = nx.get_edge_attributes(MST,'weight')
        nx.draw_networkx_edge_labels(MST,temp_pos,edge_labels=labels)
        plt.savefig("PrimMST_with_Weights.png")
        plt.close()
 
    def minKey(self, key, mstSet):
        min = sys.maxsize
 
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
 
        return min_index
 
    def primMST(self):
 
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        mstSet = [False] * self.V
        parent[0] = -1 
        key[0] = self.S

        for cout in range(self.V):
            u = self.minKey(key, mstSet)
 
            mstSet[u] = True
 
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u
 
        self.printMST(parent)

def DiGraphFileReader(file):
    next(file)
    next(file)

    vertices = int(file.readline())
    #print("Number of vertices: " + str(vertices))

    graphMatrix = [[0 for x in range(vertices)] for y in range(vertices)]
    graphMatrix2 = [[0 for x in range(vertices)] for y in range(vertices)]

    #print("Vertex\tx\t\ty")
    next(file)
    for i in range(vertices):
        xy_pos = [float(x) for x in file.readline().split()]
        #print(str(int(xy_pos[0])) + "\t" + str(xy_pos[1]) + "\t" + str(xy_pos[2]))
        G.add_node(int(xy_pos[0]), pos=(xy_pos[1], xy_pos[2]))
        DG.add_node(int(xy_pos[0]), pos=(xy_pos[1], xy_pos[2])) # reading nodes and their xy positions into the graph
    next(file)

    #print("\nEdges\t\tWeights")

    for line in file:
        if line == "\n":
            break

        numbers_float = [float(x) for x in line.split()]
        
        for j in range(1, len(numbers_float), 4):
            if int(numbers_float[0]) == int(numbers_float[j]):
                continue
            
            graphMatrix2[int(numbers_float[0])][int(numbers_float[j])] = numbers_float[j+2]
            graphMatrix[int(numbers_float[0])][int(numbers_float[j])] = numbers_float[j+2]
            DG.add_edge(int(numbers_float[0]), int(numbers_float[j]), weight = numbers_float[j+2]/10000000)

    for i in range(vertices):
        for j in range(vertices):
            if graphMatrix[i][j] != 0 and graphMatrix[j][i] == 0:
                graphMatrix[j][i] = graphMatrix[i][j]

    for i in range(vertices):
        for j in range(vertices):
            if graphMatrix2[i][j] != 0 and graphMatrix2[j][i] != 0:
                graphMatrix2[i][j] = min(graphMatrix2[i][j], graphMatrix2[j][i])
                graphMatrix2[j][i] = graphMatrix2[i][j]

            if graphMatrix2[i][j] != 0 and graphMatrix2[j][i] == 0:
                graphMatrix2[j][i] = graphMatrix2[i][j]

    for i in range(vertices):
        for j in range(i):
            if graphMatrix2[i][j] != 0:
                G.add_edge(i, j, weight = graphMatrix2[i][j]/10000000)

    for i in range(vertices):
        for j in range(vertices):
            if graphMatrix[i][j] != 0:
                DG.add_edge(j, i, weight = graphMatrix[i][j]/10000000)

    start = int(file.readline())
    #print("\nStart from: " + str(start))

    return vertices, start, graphMatrix, graphMatrix2

def GraphFileReader(file):
    next(file)
    next(file)

    vertices = int(file.readline())
    #print("Number of vertices: " + str(vertices))

    graphMatrix = [[0 for x in range(vertices)] for y in range(vertices)]

    #print("Vertex\tx\t\ty")
    next(file)
    for i in range(vertices):
        xy_pos = [float(x) for x in file.readline().split()]
        #print(str(int(xy_pos[0])) + "\t" + str(xy_pos[1]) + "\t" + str(xy_pos[2]))
        G.add_node(int(xy_pos[0]), pos=(xy_pos[1], xy_pos[2])) # reading nodes and their xy positions into the graph
    next(file)

    #print("\nEdges\t\tWeights")

    for line in file:
        if line == "\n":
            break

        numbers_float = [float(x) for x in line.split()]
        
        for j in range(1, len(numbers_float), 4):
            if int(numbers_float[0]) == int(numbers_float[j]):
                continue
            
            #print(str(int(numbers_float[0])) + "-" + str(int(numbers_float[j])) + "\t\t" + str(numbers_float[j+2]/10000000))
            
            graphMatrix[int(numbers_float[0])][int(numbers_float[j])] = numbers_float[j+2]
            # G.add_edge(int(numbers_float[0]), int(numbers_float[j]), weight = numbers_float[j+2]/10000000)

    for i in range(vertices):
        for j in range(vertices):
            if graphMatrix[i][j] != 0 and graphMatrix[j][i] != 0:
                graphMatrix[i][j] = min(graphMatrix[i][j], graphMatrix[j][i])
                graphMatrix[j][i] = graphMatrix[i][j]

            if graphMatrix[i][j] != 0 and graphMatrix[j][i] == 0:
                graphMatrix[j][i] = graphMatrix[i][j]

    for i in range(vertices):
        for j in range(i):
            if graphMatrix[i][j] != 0:
                G.add_edge(i, j, weight = graphMatrix[i][j]/10000000)

    start = int(file.readline())
    #print("\nStart from: " + str(start))

    return vertices, start, graphMatrix


def readInputFile(filename):
    #filename = "input100.txt"
    f = open(filename, "r")
    
    # for algorithms other than prims/kruskal/clustering
    global DG, G
    DG = nx.DiGraph()
    G = nx.Graph()
    global di_verts
    global verts
    global di_starting
    global starting
    global digraphMat
    global graphMat
    di_verts, di_starting, digraphMat, graphMat = DiGraphFileReader(f)
    pos=nx.get_node_attributes(DG,'pos')
    nx.draw(DG, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.savefig("DiGraph.png")
    labels = nx.get_edge_attributes(DG,'weight')
    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)
    plt.savefig("DiGraph_with_Weights.png")
    f.close()
    plt.close()

    # for prims/kruskal/clustering
    # f = open(filename, "r")
    # global G
    # G = nx.Graph()
    # global verts
    # global starting
    # global graphMat
    # verts, starting, graphMat = GraphFileReader(f)
    verts = di_verts
    starting = di_starting
    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.savefig("Graph.png")
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.savefig("Graph_with_Weights.png")
    # f.close()
    plt.close()

    # primG = Graph(verts, starting)
    # primG.graph = graphMat
    # primG.primMST()

def PrimAlgo():
    primG = PrimGraph(verts, starting)
    primG.graph = graphMat
    primG.primMST()

    return primG.getTotalWeight()

def KruskalAlgo():
    kruskalG = KruskalGraph(verts, starting)
    cost_Mat = [[float('inf') for x in range(verts)] for y in range(verts)]
    
    for i in range(verts):
        for j in range(verts):
            if graphMat[i][j] != 0:
                cost_Mat[i][j] = graphMat[i][j]

    kruskalG.graph = cost_Mat
    kruskalG.kruskalMST()

    return kruskalG.getTotalWeight()

def DijkstraAlgo():
    dijkstraG = DijkstraGraph()
    dijkstraG.dijkstra(digraphMat, starting)

def BellmanFordAlgo():
    dist = [float('Inf')]*di_verts
    dist[starting] = 0
    BellmanFordGraph = nx.DiGraph()
    BellmanFordPositions = nx.get_node_attributes(DG, "pos")
    
    cost_Mat = [[float('Inf') for x in range(verts)] for y in range(verts)]
    
    for i in range(di_verts):
        # bf_x, bf_y = BellmanFordPositions[i]
        BellmanFordGraph.add_node(i, pos = BellmanFordPositions[i])
        for j in range(di_verts):
            if digraphMat[i][j] != 0:
                cost_Mat[i][j] = digraphMat[i][j]/10000000

    for some in range(di_verts-1):
        for u in range(di_verts):
            for v in range(di_verts):
                if cost_Mat[u][v] != float('Inf'):
                    if dist[u] != float('Inf') and dist[u] + cost_Mat[u][v] < dist[v]:
                        BellmanFordGraph.add_edge(u, v, weight = cost_Mat[u][v])
                        dist[v] = dist[u] + cost_Mat[u][v]
                        
    

    for u in range(di_verts):
        for v in range(di_verts):
            if dist[u] != float('Inf') and dist[u] + cost_Mat[u][v] < dist[v]:
                print("Graph contains cycle.")
                return

    # for i in range(di_verts):
    #         print("{0}\t\t{1}".format(i, dist[i]))

    nx.draw(BellmanFordGraph, BellmanFordPositions, with_labels=True)
    plt.savefig("BellmanFordGraph.png")
    labels = nx.get_edge_attributes(BellmanFordGraph,'weight')
    nx.draw_networkx_edge_labels(BellmanFordGraph,BellmanFordPositions,edge_labels=labels)
    plt.savefig("BellmanFordGraph_with_Weights.png")
    plt.close()

# def BellmanFordAlgo():
#     dist = [float('Inf')]*verts
#     dist[starting] = 0
#     BellmanFordGraph = nx.DiGraph()
#     BellmanFordPositions = nx.get_node_attributes()
    
#     cost_Mat = [[float('Inf') for x in range(verts)] for y in range(verts)]
#     for i in range(verts):
#         for j in range(verts):
#             if graphMat[i][j] != 0:
#                 cost_Mat[i][j] = graphMat[i][j]/10000000
#                 BellmanFordGraph.add_node(i, j, weight = cost_Mat[i][j])

#     for some in range(verts-1):
#         for u in range(verts):
#             for v in range(verts):
#                 if cost_Mat[u][v] != float('Inf'):
#                     if dist[u] != float('Inf') and dist[u] + cost_Mat[u][v] < dist[v]:

#                         dist[v] = dist[u] + cost_Mat[u][v]

#     for u in range(verts):
#         for v in range( verts):
#             if dist[u] != float('Inf') and dist[u] + cost_Mat[u][v] < dist[v]:
#                 print("Graph contains cycle bitch")
#                 return

#     for i in range(di_verts):
#             print("{0}\t\t{1}".format(i, dist[i]))

def FloydWarshallAlgo():
    cost_Mat = [[float('inf') for x in range(verts)] for y in range(verts)]
    
    for i in range(verts):
        for j in range(verts):
            if digraphMat[i][j] != 0:
                cost_Mat[i][j] = digraphMat[i][j]/10000000

    floydW = FloydWarshall(verts, cost_Mat)

    floydW.MakeGraph("FloydWarshall_Before")
    resultantMatrix = floydW.algo()
    floydW.MakeGraph("FloydWarshall_After")

    return resultantMatrix

def ClusteringCoefficientAlgo():
    local_cluster = {}
    for node in G:

        adjacentNodes = []
        indirectly_connected_nodes = []

        for n in G.neighbors(node):
            adjacentNodes.append(n)

        for n in G.neighbors(node):
            for n2 in G.neighbors(n):
                if n2 in adjacentNodes:
                    indirectly_connected_nodes.append(n2)

        indirectly_connected_nodes = list(indirectly_connected_nodes)

        cluster = 0
        if len(indirectly_connected_nodes):
            cluster =  (float(len(indirectly_connected_nodes)))/((float(len(list(G.neighbors(node)))) * (float(len(list(G.neighbors(node)))) - 1)))

        local_cluster[node] = cluster

    # print("Node:\tCoefficient:")
    avg_coefficient = 0
    for i in range(len(local_cluster)):
        # print(str(i) + "\t" + str(local_cluster[i]))
        avg_coefficient += local_cluster[i]
    avg_coefficient /= len(local_cluster)
    # print(avg_coefficient)

    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
    min_x = float('inf')
    min_y = float('inf')

    for i in range(len(local_cluster)):
        x,y = pos[i]

        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

        plt.text(x,y+0.03,s=str(round(local_cluster[i], 2)), bbox=dict(facecolor='red'),horizontalalignment='center')
    
    plt.text(min_x, min_y-0.009, 'Average Coefficient = ' + str(round(avg_coefficient, 3)), fontsize = 22, horizontalalignment= "left", verticalalignment = "top")
    plt.savefig("ClusteringCoefficient.png")
    #plt.show()
    plt.close()

    return local_cluster

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else :
        parent[yroot] = xroot
        rank[xroot] += 1

def BoruvkaAlgo():
    adjlist = []
    parent = []
    rank = []
    cheapest = []
    BoruvkaTotalWeight = 0
    numTree = verts
    BoruvkaGraph = nx.Graph()
    BoruvkaPositions = nx.get_node_attributes(G, "pos")

    for i in range(verts):
        for j in range(verts):
            if graphMat[i][j] != 0:
                adjlist.append([i, j, graphMat[i][j]/10000000])

    for vertex in range(verts):
        parent.append(vertex)
        rank.append(0)
        cheapest = [-1] * verts
        BoruvkaGraph.add_node(vertex, pos = BoruvkaPositions[vertex])

    while numTree > 1:
        for i in range(len(adjlist)):
            u, v, w = adjlist[i]
            set1 = find(parent, u)
            set2 = find(parent, v)

            if set1 != set2:
                if cheapest[set1] == -1 or cheapest[set1][2] > w :
                    cheapest[set1] = [u,v,w] 
  
                if cheapest[set2] == -1 or cheapest[set2][2] > w :
                    cheapest[set2] = [u,v,w]

        for node in range(verts):
            #Check if cheapest for current set exists
            if cheapest[node] != -1:
                u,v,w = cheapest[node]
                set1 = find(parent, u)
                set2 = find(parent ,v)
  
                if set1 != set2 :
                    BoruvkaTotalWeight += w
                    union(parent, rank, set1, set2)
                    # print ("Edge %d-%d with weight %d included in MST" % (u,v,w))
                    BoruvkaGraph.add_edge(u, v, weight = w)
                    numTree = numTree - 1

        cheapest = [-1] * verts
    
    min_x = min_y = float('inf')
    for i in range(BoruvkaGraph.number_of_nodes()):
        x, y = BoruvkaPositions[i]
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
    
    nx.draw(BoruvkaGraph, BoruvkaPositions, with_labels=True)
    plt.text(min_x, min_y-0.012, 'MST = ' + str(round(BoruvkaTotalWeight, 2)), fontsize = 22, horizontalalignment = "left", verticalalignment = "top")
    plt.savefig("BoruvkaGraph.png")
    labels = nx.get_edge_attributes(BoruvkaGraph, 'weight')
    nx.draw_networkx_edge_labels(BoruvkaGraph, BoruvkaPositions, edge_labels=labels)
    plt.savefig("BoruvkaGraph_with_Weights.png")
    plt.close()

# readInputFile("Input10.txt")
# BellmanFordAlgo()
# DijkstraAlgo()
# BoruvkaAlgo()