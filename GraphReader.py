import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import sys

class PrimGraph():
 
    def __init__(self, Vertices, Starting_Vert):
        self.V = Vertices
        self.S = Starting_Vert
        self.TotalWeight = 0
        self.graph = [[0 for i in range(Vertices)] for j in range(Vertices)]

    def getTotalWeight(self):
        return self.TotalWeight
 
    def printMST(self, parent):
        # print("Edge \tWeight")
        # for i in range(1, self.V):
        #     print(parent[i], "-", i, "\t", self.graph[i][ parent[i] ])

        temp_pos = nx.get_node_attributes(G, 'pos')
        MST = nx.Graph()
        for i in range(self.V):
            MST.add_node(i, pos = temp_pos[i])

        for i in range(1, self.V):
            MST.add_edge(parent[i], i, weight = self.graph[i][parent[i]]/10000000)
            self.TotalWeight += self.graph[i][parent[i]]/10000000

        MSTpos=nx.get_node_attributes(MST,'pos')
        nx.draw(MST, MSTpos, with_labels=True, connectionstyle="arc3,rad=0.1")
        plt.title("MST Total Weight: " + str(self.TotalWeight))
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

def KruskalAlgo():
    pass

def DijkstraAlgo():
    pass

def BellmanFordAlgo():
    pass

def FloydWarshallAlgo():
    pass

def ClusteringCoefficientAlgo():
    pass

def BoruvkaAlgo():
    pass