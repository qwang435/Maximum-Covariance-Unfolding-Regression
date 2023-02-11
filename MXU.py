import sys
from mosek.fusion import *
import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

def MCU(X,Y,n_neigh):

    with Model("sdo1") as M:

        A=np.matmul(X,X.T)
        P=np.matmul(Y,Y.T)
        n = np.size(X,0)

        # Calculate the nearest neighbors of each data point and build a graph
        #N = NearestNeighbors(n_neighbors=n_neigh).fit(Y).kneighbors_graph(Y).todense()
        #N = np.array(N)
        N = kneighbors_graph(Y, n_neigh).toarray()
        N = np.maximum(N,N.T)


        # Setting up the variables
        Q = M.variable("Q", Domain.inPSDCone(n))

        # Setting up constant coefficient matrices
        C = Matrix.dense(A)
        Ones = Matrix.ones(n,n)
        Eye = Matrix.eye(n)

        # Objective
        M.objective(ObjectiveSense.Maximize, Expr.dot(C, Q))

        # Constraints
        M.constraint("c0", Expr.dot(Ones, Q), Domain.equalsTo(0))
        M.constraint("c1", Expr.dot(Eye, Q), Domain.lessThan(100000000*n*(n-1)))

        count = 0
        for i in range(n):
            for j in range(n):
                if N[i, j] == 1 and i<j:
                    count += 1
                    M.constraint(Expr.add(Expr.add(Q.index([i,i]), Q.index([j,j])),Expr.mul(-2,Q.index([i,j]))), Domain.equalsTo(P[i,i]+P[j,j]-2*P[i,j]))
        print(count)

        # Solve
        M.setLogHandler(sys.stdout)
        M.acceptedSolutionStatus(AccSolutionStatus.Optimal)
        M.solve()

        return getYtilda(np.reshape(Q.level(),(n,n)))

def MVU(Y,n_neigh):

    with Model("sdo1") as M:

        P = np.matmul(Y,Y.T)
        n = np.size(Y,0)

        # Calculate the nearest neighbors of each data point and build a graph
        N = kneighbors_graph(Y, n_neigh).toarray()
        N = np.maximum(N,N.T)

        # Setting up the variables
        Q = M.variable("Q", Domain.inPSDCone(n))

        # Setting up constant coefficient matrices
        Ones = Matrix.ones(n,n)
        Eye = Matrix.eye(n)

        # Objective
        M.objective(ObjectiveSense.Maximize, Expr.dot(Eye, Q))

        # Constraints
        M.constraint("c0", Expr.dot(Ones, Q), Domain.equalsTo(0.0))
        M.constraint("c1", Expr.dot(Eye, Q), Domain.lessThan(100000*n*(n-1)))

        count = 0
        for i in range(n):
            for j in range(n):
                if N[i, j] == 1 and i<j:
                    count += 1
                    M.constraint(Expr.add(Expr.add(Q.index([i,i]), Q.index([j,j])),Expr.mul(-2,Q.index([i,j]))), Domain.equalsTo(P[i,i]+P[j,j]-2*P[i,j]))
        print(count)

        # Solve
        M.setLogHandler(sys.stdout)
        M.acceptedSolutionStatus(AccSolutionStatus.Optimal)
        M.solve()

        return getYtilda(np.reshape(Q.level(),(n,n)))

def getYtilda(Q):

    [evals, V] = np.linalg.eigh(Q)
    evals = evals[::-1]
    V = np.flip(V, axis=1)
    Y_tilda = V.dot(np.diag(np.sqrt(np.abs(evals))))

    return Y_tilda
