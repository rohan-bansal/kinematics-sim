import osqp as osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import time

def initial_test_run():
    DELTA_T = 0.1
    N = 10

    umin = np.array([-20, -20])
    umax = np.array([20, 20])
    xmin = np.array([-30, -30, -5, -5])
    xmax = np.array([30, 30, 5, 5])
    x0 = np.array([2, 3, 2, 3])
    xr = np.array([2.5, 3.5, 0, 0])

    Q = sparse.eye(4)
    R = sparse.eye(2) * 0.01

    nx, nu = [4, 2]
    QN = Q

    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)], format='csc')

    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -1 * Q.dot(xr)), -1 * QN.dot(xr),
                np.zeros(N*nu)])

    Ad = sparse.csc_matrix([[1, 0, DELTA_T, 0], [0, 1, 0, DELTA_T], [0, 0, 1, 0], [0, 0, 0, 1]])
    Bd = sparse.csc_matrix([[0, 0], [0, 0], [DELTA_T, 0], [0, DELTA_T]])

    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(N*nx)])
    ueq = leq
    # - input and state constraints
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])


    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, warm_start=True)

    res = prob.solve()
    # print(res, res.x, res.x[N*(nx-1):N*nx])
    x_pos = res.x[0:N*nx:4]
    y_pos = res.x[1:N*nx:4]

    print(x_pos, y_pos, res.x)

    fig, ax = plt.subplots()
    


    ax.plot(x_pos, y_pos)
    ax.scatter(x_pos, y_pos)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    # for i, txt in enumerate(labels):
    #     ax.text(
    #         x_pos[i], y_pos[i], txt, ha="right", va="bottom", fontsize=5
    #     )  # adjust positioning as needed

    plt.show()

def randomized_test_run_return_time(num_tests, numRandomPlotsShown):
    totalTime = 0
    numPlotsDone = 0
    for i in range(0, num_tests):
        DELTA_T = 0.5
        N = 10 
        umin = np.array([-np.random.randint(20, 30), -np.random.randint(20, 30)])
        umax = np.array([np.random.randint(20, 30), np.random.randint(20, 30)])
        xmin = np.array([-np.random.randint(10, 30), -np.random.randint(10, 30), -np.random.randint(1, 20), -np.random.randint(1, 20)])
        xmax = np.array([np.random.randint(10, 30), np.random.randint(10, 30), np.random.randint(1, 20), np.random.randint(1, 20)])
        x0 = np.array([np.random.randint(xmin[0] + 5, xmax[0] - 5), np.random.randint(xmin[1] + 5, xmax[1] - 5), np.random.randint(xmin[2], xmax[2]), np.random.randint(xmin[3], xmax[3])])
        xr = np.array([np.random.randint(xmin[0] + 5, xmax[0] - 5), np.random.randint(xmin[1] + 5, xmax[1] - 5), np.random.randint(xmin[2], xmax[2]), np.random.randint(xmin[3], xmax[3])])

        Q = sparse.eye(4)
        R = np.eye(2) * np.random.random() * 1

        nx, nu = [4, 2]
        QN = Q

        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')

        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -1 * Q.dot(xr)), -1 * QN.dot(xr),
                    np.zeros(N*nu)])

        Ad = sparse.csc_matrix([[1, 0, DELTA_T, 0], [0, 1, 0, DELTA_T], [0, 0, 1, 0], [0, 0, 0, 1]])
        Bd = sparse.csc_matrix([[0, 0], [0, 0], [DELTA_T, 0], [0, DELTA_T]])

        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])


        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True)

        startTime = time.time()
        res = prob.solve()
        endTime = time.time()

        totalTime += (endTime - startTime)
        # print(res, res.x, res.x[N*(nx-1):N*nx])
        x_pos = res.x[0:N*nx:4]
        y_pos = res.x[1:N*nx:4]

        if numPlotsDone < numRandomPlotsShown:
            fig, ax = plt.subplots()
            ax.plot(x_pos, y_pos)
            ax.scatter(x_pos, y_pos)

            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            numPlotsDone += 1

            # for i, txt in enumerate(labels):
            #     ax.text(
            #         x_pos[i], y_pos[i], txt, ha="right", va="bottom", fontsize=5
            #     )  # adjust positioning as needed

            plt.show()
    return totalTime

# initial_test_run()
def graphTime(START_PROBS, MAX_PROBS, increment=1):
    dataPoints50 = []
    num_tests = [1] + [increment * i for i in range(int(START_PROBS/increment), 1 + int(MAX_PROBS/increment))]

    for i in range(0, len(num_tests)):
        times = []
        for x in range(1):
            t = randomized_test_run_return_time(num_tests[i], 0)
            times.append(t)
        t_ave = np.average(times)
        dataPoints50.append(t_ave)
        print(str(num_tests[i]) + " problems")
        print(str(t_ave) + " seconds")
        print()
        print()

    plt.plot(num_tests, dataPoints50)
    plt.scatter(num_tests, dataPoints50)

    plt.xlabel('Number of Quadratic Problems')
    plt.ylabel('Time to Solve (Seconds)')
    plt.title('OSQP')

    plt.show()

graphTime(1, 1000, increment=10)