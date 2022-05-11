import numpy as np

class Graph():
    """
    example data
        vertices = {'a', 'b', 'c', ...}
        edges = {'a' = {'b', 'c'}, ...}
    """

    def __init__(self, vertices = [], edges=[]):
        self.vertices = []
        self.edges = {}
        for v in vertices:
            self.addVertice(v)
        for e in edges:
            self.addEdges()

    def addVertice(self, vertice):
        self.vertices.append(vertice)
        self.edges[vertice] = []
    
    def addEdges(self, edges):
        for e in edges:
            self.edges[e[0]].append(e[1])
            self.edges[e[1]].append(e[0])
    
    def getVertices(self) -> list:
        return self.vertices

    def getEdges(self) -> dict:
        return self.edges
    
    def getConnectivityMatrix(self):
        vertices = self.getVertices()
        matrix = []
        for v in vertices:
            row = np.zeros(len(vertices))
            for e in self.edges[v]:
                row[vertices.index(e)]=1
            matrix.append(row)
        return matrix