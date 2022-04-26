import numpy as np

class Graph():

    """
    example data
        vertices = {'a', 'b', 'c', ...}
        edges = {'a' = {'b', 'c'}, ...}
    """
    def __init(self, vertices = set(), edges={}):
        for v in vertices:
            self.addVertice(v)
        for e in edges:
            self.addEdges()


    def addVertice(self, vertice):
        self.vertices.add(vertice)
        self.edges[vertice] = []
    
    def addEdges(self, edges):
        for e in edges:
            self.edges[e[0]].append(e[1])
            self.edges[e[1]].append(e[0])
    
    def getVertices(self) -> list:
        return list(self.vertices)

    def getEdges(self) -> dict:
        return self.edges

    
    def getConnectivityMatrix(self):
        sorted_set = sorted(self.vertices)
        matrix = []
        for v in sorted_set:
            row = np.zeros(len(sorted_set))
            for e in self.edges[v]:
                row[sorted_set.index(e)]=1
            matrix.append(row)
        return matrix
    
    