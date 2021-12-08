import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import sys

from networkx.algorithms.cluster import average_clustering

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

    #print("Vertex\tx\t\ty")
    next(file)
    for i in range(vertices):
        xy_pos = [float(x) for x in file.readline().split()]
        #print(str(int(xy_pos[0])) + "\t" + str(xy_pos[1]) + "\t" + str(xy_pos[2]))
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
            
            #print(str(int(numbers_float[0])) + "->" + str(int(numbers_float[j])) + "\t\t" + str(numbers_float[j+2]/10000000))
            graphMatrix[int(numbers_float[0])][int(numbers_float[j])] = numbers_float[j+2]
            DG.add_edge(int(numbers_float[0]), int(numbers_float[j]), weight = numbers_float[j+2]/10000000)

    for i in range(vertices):
        for j in range(vertices):
            if graphMatrix[i][j] != 0 and graphMatrix[j][i] == 0:
                graphMatrix[j][i] = graphMatrix[i][j]
                DG.add_edge(j, i, weight = graphMatrix[i][j]/10000000)

    start = int(file.readline())
    #print("\nStart from: " + str(start))

    return vertices, start, graphMatrix

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
    global DG
    DG = nx.DiGraph()
    global di_verts
    global di_starting
    global digraphMat
    di_verts, di_starting, digraphMat = DiGraphFileReader(f)
    pos=nx.get_node_attributes(DG,'pos')
    nx.draw(DG, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.savefig("DiGraph.png")
    labels = nx.get_edge_attributes(DG,'weight')
    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)
    plt.savefig("DiGraph_with_Weights.png")
    f.close()
    plt.close()

    # for prims/kruskal/clustering
    f = open(filename, "r")
    global G
    G = nx.Graph()
    global verts
    global starting
    global graphMat
    verts, starting, graphMat = GraphFileReader(f)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.savefig("Graph.png")
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.savefig("Graph_with_Weights.png")
    f.close()
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
    pass

def BellmanFordAlgo():
    pass

def FloydWarshallAlgo():
    pass

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

    return local_cluster, avg_coefficient

def BoruvkaAlgo():
    pass