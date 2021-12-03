import numpy as np


def feature_extract(node_graph):
    val_node = node_graph[0].copy()
    graph = node_graph[1].copy()
    n_nodes = len(graph)
    m = int(n_nodes / 2)
    dis_mat = []
    for n_node in range(n_nodes):
        new_mat = np.zeros([n_nodes])
        matrix = [graph.copy(), graph.T.copy()]
        for idx, mat in enumerate(matrix):
            bfs = []
            tmp1_node = [n_node]
            tmp2_node = []
            # begin
            while len(tmp1_node) + len(tmp2_node) > 0:
                # for each parent
                for parent_node in tmp1_node:
                    tmp = [parent_node, []]
                    # for each child, row
                    for child_node, child_weight in enumerate(mat[parent_node, :]):
                        if child_weight > 0:
                            # add current node info
                            tmp[1].append(child_node)
                            # add children to tmp2
                            tmp2_node.append(child_node)
                            # remove this edge
                            mat[parent_node, child_node] = 0

                    bfs.append(tmp.copy())
                # to next level
                tmp1_node = tmp2_node.copy()
                tmp2_node = []
            # print(bfs)
            keys = []
            values = []
            for i in range(len(bfs)):
                if bfs[i][0] not in keys:
                    keys.append(bfs[i][0])
                    values.append(bfs[i][1])
            bfs_dict = dict(zip(keys, values))
            # print(bfs_dict)
            if n_node in val_node:
                if idx == 0:
                    new_mat[0] = 1
                    new_mat[2] = len(bfs_dict[n_node]) + 1
                else:
                    new_mat[1] = 1
                    new_mat[3] = len(bfs_dict[n_node]) + 1
            for i in range(m - 2):
                adj_node = []
                queue = [n_node]
                if n_node in val_node:
                    adj_node.append(n_node)
                adj_node.extend(bfs_dict[n_node])
                for loop in range(i + 1):
                    a = []
                    for q_node in queue:

                        a.extend(bfs_dict[q_node])
                        for node in bfs_dict[q_node]:
                            # print(node)
                            for b in bfs_dict[node]:
                                # print(b)
                                if b not in adj_node:
                                    adj_node.append(b)
                    queue = a.copy()

                # print(adj_node)
                # print(len(adj_node))
                if idx == 0:
                    new_mat[2 * (i + 2)] = len(adj_node)
                else:
                    new_mat[2 * (i + 2) + 1] = len(adj_node)

        dis_mat.append(new_mat)

    diag_mat = np.zeros([n_nodes, n_nodes])
    for v_node in val_node:
        diag_mat[v_node, v_node] = 1

    Lap_mat = np.zeros([n_nodes, n_nodes])
    Lap_mat[range(n_nodes), range(n_nodes)] = np.sum(graph, 1)
    Lap_mat = Lap_mat - graph

   # ft_mat = np.zeros((3, n_nodes, n_nodes))
   # ft_mat[0, :, :] = diag_mat  # diagnol matrix
   # ft_mat[1, :, :] = np.vstack(dis_mat)  # nodes matrix
    #ft_mat[2, :, :] = Lap_mat  # Laguassian matrix


    return diag_mat,Lap_mat,np.vstack(dis_mat)