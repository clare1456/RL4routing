# applying gurobipy to solve VRPTW

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import time

import GraphTool

class ModelHandler():
    def __init__(self, graph):
        self.graph = graph

    def build_VRPTW(self):
        """
        VRPTW model in default
        """
        # building model
        model = gp.Model('VRPTW')

        nodeNum = self.graph.nodeNum
        points = list(range(nodeNum))
        A = [(i, j) for i in points for j in self.graph.feasibleNodeSet[i]]

        ## add variates
        x = model.addVars(A, vtype=GRB.BINARY, name="x")
        t = model.addVars(points, vtype=GRB.CONTINUOUS, name="t")
        q = model.addVars(points, vtype=GRB.CONTINUOUS, name="q")
        ## set objective
        model.modelSense = GRB.MINIMIZE
        model.setObjective(gp.quicksum(x[i, j] * self.graph.disMatrix[i, j] for i, j in A))
        ## set constraints
        ### 1. flow balance
        model.addConstrs(gp.quicksum(x[i, j] for j in self.graph.feasibleNodeSet[i] if j!=i)==1 for i in points[1:]) # depot not included
        model.addConstrs(gp.quicksum(x[i, j] for i in self.graph.availableNodeSet[j] if i!=j)==1 for j in points[1:]) # depot not included
        ### 2. avoid subring / self-loop
        model.addConstrs((x[i, j] == 1) >> (t[j] >= t[i] + self.graph.serviceTime[i] + self.graph.timeMatrix[i, j]) for i, j in A if j!=0)
        ### 3. time constraints
        model.addConstrs(t[i] >= self.graph.readyTime[i] for i in points)
        model.addConstrs(t[i] <= self.graph.dueTime[i] for i in points)
        ### 4. capacity constraints
        model.addConstrs((x[i, j] == 1) >> (q[j] >= q[i] + self.graph.demand[i]) for i, j in A if j!=0)
        model.addConstrs(q[i] <= self.graph.capacity for i in points)
        model.addConstrs(q[i] >= 0 for i in points)
        ### 5. vehicle number constraint
        model.addConstr(gp.quicksum(x[0, j] for j in self.graph.feasibleNodeSet[0]) <= self.graph.vehicleNum)

        # update model
        model.update()

        self.model = model
        return model

    def build_TSP(self):
        """
        TSP model
        """
        # building model
        model = gp.Model('TSP')
        self.graph.feasibleNodeSet = [list(range(self.graph.nodeNum)) for _ in range(self.graph.nodeNum)]
        self.graph.availableNodeSet = [list(range(self.graph.nodeNum)) for _ in range(self.graph.nodeNum)]

        nodeNum = self.graph.nodeNum
        points = list(range(nodeNum))
        A = [(i, j) for i in points for j in points]

        ## add variates
        x = model.addVars(A, vtype=GRB.BINARY, name="x")
        u = model.addVars(points, vtype=GRB.INTEGER, name="u")
        ## set objective
        model.modelSense = GRB.MINIMIZE
        model.setObjective(gp.quicksum(x[i, j] * self.graph.disMatrix[i, j] for i, j in A))
        ## set constraints
        ### 1. flow balance
        model.addConstrs(gp.quicksum(x[i, j] for j in points if j!=i)==1 for i in points) # depot not included
        model.addConstrs(gp.quicksum(x[i, j] for i in points if i!=j)==1 for j in points) # depot not included
        ### 2. avoid subring / self-loop
        model.addConstrs((x[i, j] == 1) >> (u[j] >= u[i] + 1) for i, j in A if j!=0)

        # update model
        model.update()

        self.model = model
        return model

    def get_routes(self, model=None):
        """
        get routes from model
        """
        if model is None:
            model = self.model
        # get the routes
        routes = []
        for j in self.graph.feasibleNodeSet[0]:
            if round(model.getVarByName(f"x[0,{j}]").X) == 1:
                route = [0]
                route.append(j)
                i = j
                while j != 0:
                    for j in self.graph.feasibleNodeSet[i]:
                        if round(model.getVarByName(f"x[{i},{j}]").X) == 1:
                            route.append(j)
                            i = j
                            break
                routes.append(route)
        return routes

    def get_obj(self, model=None):
        """
        get optimal objective value of model
        """
        if model is None:
            model = self.model
        return model.ObjVal

    def draw_routes(self, routes=None):
        """
        draw routes
        """
        if routes is None:
            routes = self.get_routes()
        self.graph.render(routes)

    def run(self):
        """
        solve model with gurobi solver 
        """
        # build model
        self.build_TSP()
        # self.build_VRPTW()
        # optimize the model
        self.model.optimize()
        # return routes
        if self.model.status == 2:
            return self.get_routes()
        else:
            print("Failed: Model is infeasible")
            return []

if __name__ == "__main__":
    # solve model with gurobi solver
    file_name = "solomon_100/R101.txt"
    # 101\102-1s, 101_25-0.02s, 103-200s
    graph = GraphTool.Graph(file_name, limit_node_num=10)
    alg = ModelHandler(graph) #! TSP model
    time1 = time.time()
    routes = alg.run()
    time2 = time.time()
    for ri in range(len(routes)):
        print("route {}: {}".format(ri, routes[ri]))
    info = {}
    graph.evaluate(routes, info=info)
    graph.render(routes)
    print("optimal obj: {}\ntime consumption: {}".format(graph.evaluate(routes), time2-time1))

