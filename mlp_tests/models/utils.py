import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

def sample_nsphere(n):
    angles = np.random.normal(0, 1, n)
    angles_norm = np.linalg.norm(angles, 2)
    nsphere_sample = angles / angles_norm
    return nsphere_sample


def generate_random_basis(dim=3, allow_vertical=False, seed=42):
    np.random.seed(seed)

    # Generate random basis for hyperplane in d-dim space. Time complexity O(d^2).
    # Algorithm Monte-Carlo Graham Smith Orthogonalization
    m_v = []
    up_vec = np.zeros(dim)
    up_vec[dim - 1] = 1
    for i in range(0, dim):
        ei = sample_nsphere(dim)

        while True:
            is_vertical = math.isclose(0.8, np.abs(np.dot(ei, up_vec)))  # Avoid close-to-vertical hyperplane
            if allow_vertical or not is_vertical:
                is_orthogonalizable = True  # Graham Smith Orthogonalization
                for j in range(0, i):
                    proj = np.dot(ei, m_v[j])*m_v[j]
                    diff = ei - proj
                    ei = diff
                    if math.isclose(0, np.linalg.norm(ei, ord=2)):
                        is_orthogonalizable = False
                        break
                    norm = ei / np.linalg.norm(ei, ord=2)
                    ei = norm
                if is_orthogonalizable is True:
                    break
            ei = sample_nsphere(dim)
        m_v.append(ei)

    return np.array(m_v).transpose()

if __name__=="__main__":
    hyperplane_d_1 = np.random.uniform(-5, 5, (30, 3))
    hyperplane_d_1[0:, 2] = 0

    m_v = generate_random_basis(3, seed=3288)
    m_e = np.linalg.inv(m_v)

    transformed = np.matmul(hyperplane_d_1, m_e.transpose())
    #transformed = transformed + np.array([3, 5, 0])

    p0 = m_v[0]
    p1 = m_v[1]
    p2 = m_v[2]

    origin = [0, 0, 0]
    X, Y, Z = zip(origin, origin, origin)
    U, V, W = zip(p0*10, p1*10, p2*10)

    normal = p2
    d = 0
    xx, yy = np.meshgrid(range(-7, 7), range(-7, 7))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z, alpha=0.2)
    plt3d.plot(transformed[0:, 0], transformed[0:, 1], transformed[0:, 2], 'go')
    plt3d.plot(hyperplane_d_1[0:, 0], hyperplane_d_1[0:, 1], hyperplane_d_1[0:, 2], 'mo')
    plt3d.quiver(X, Y, Z, U, V, W, color="red")
    plt3d.set_xlim([-5, 5])
    plt3d.set_ylim([-5, 5])
    plt3d.set_zlim([-5, 5])
    plt.show()