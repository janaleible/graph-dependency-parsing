import numpy as np



def Edmonds(A, start):
    """
    :param A: square matrix, where Root is the first entry. Row numbers correspond to the base of the edge, columns to the head
    :return: max_span, a square mtx which corresponds to the maximum spanning tree
    """


    # Deleting diagonal elements (node should not have an arc to itself) and deleting incoming edges to the Root node
    A = preprocess(A, start)

    # calculating greedy matrix (include edges with max incoming weight)
    B = max_incoming_edges(A)

    # creating list for every node which node it is connected to
    connection_list = connections(B)

    # findig circle in the greedy graph
    circle = circle_finder(B, connection_list)

    not_circle = np.arange(A.shape[0])
    not_circle = np.delete(not_circle, circle)

    # in case there are no circles we have our max_spanning tree
    if(circle == []):

        return B

    # otherwise calculate the collapsed matrix
    else:
        collapsed, rec1, rec2, expand_map = calculate_collapsed(circle, not_circle, A)

        #calling Edmonds algorithm recursively
        collapsed_max_span = Edmonds(collapsed, 0)

        # reconstructing the original
        max_span = reconstruct(collapsed_max_span, rec1, rec2, expand_map, A, circle, not_circle)

    return max_span




def reconstruct(collapsed_max_span, rec1, rec2, expand_map, A, circle, not_circle):
    reconstructed = np.zeros(A.shape)

    # maps non_circle elements to the relevant rows and columns of the matrix, fill in corresponding indices
    for i in range(len(rec1)):
        for j in range(len(rec1)):
            reconstructed[expand_map[i], expand_map[j]] = collapsed_max_span[i, j]


    # map the circle elements to their places and fill in the numbers in (c, nc) and (nc,c) places based on rec1, rec2

    # rec1 (i,j) = nc element i connects to c element j
    # there is only on incoming arc in the collapsed_max_span, find the nc position of it and copy that index and the rec1

    #first fill in the (nc,c) link
    for i in range(collapsed_max_span.shape[0]-1):
        if (collapsed_max_span[i,-1] != 0):
            nc_index = i

    reconstructed[expand_map[nc_index], circle[rec1[nc_index]]] = A[expand_map[nc_index], circle[rec1[nc_index]]]


    # reconstruct outgoing weights from collapsed node
    # check which outgoing arcs are present in the collapsed_max_span, find which node it comes from, fill reconstructed accordingly

    for i in range(collapsed_max_span.shape[1]-1):
        if collapsed_max_span[-1, i] != 0:
            reconstructed[rec2[expand_map[i]], expand_map[i]] = collapsed_max_span[-1, i]

    # (c,c) places must be filled regarding the max spanning tree, with specific root-node
    # construct matrix of circle entities
    circle = circle[:-1]

    circle_mtx = A.copy()
    circle_mtx = circle_mtx[:,circle][circle,:]

    circle_maxspan = Edmonds(circle_mtx, rec1[nc_index])

    for i in range(circle_maxspan.shape[0]):
        for j in range(circle_maxspan.shape[1]):
            reconstructed[circle[i], circle[j]] = circle_maxspan[i,j]

    return reconstructed

def calculate_collapsed(circle, not_circle, A):
    shrink_map, expand_map = calculate_mapping(not_circle)
    collapsed = collapse_circle(A, circle, not_circle)

    collapsed, reconstruct1 = fill_incoming_weights(A, collapsed, circle, not_circle)
    collapsed, reconstruct2 = fill_outgoing_weights(A, collapsed, circle, shrink_map)
    return collapsed, reconstruct1, reconstruct2, expand_map

def preprocess(A, i):
    dim = A.shape[0]
    # setting diagonal elements to 0 (we do not need loop edges)
    A = A - np.diag(np.diag(A))
    A[:, i] = np.zeros(dim)
    return(A)

def max_incoming_edges(A):
    dim = A.shape[0]


    max_ind = np.argmax(A,0)
    max_val = np.amax(A,0)

    B = np.zeros(A.shape)
    for i in range(dim):
        B[max_ind[i],i] = max_val[i]
    return B


def connections(A):
    dim = A.shape[0]

    #creating the list of possible connections
    can_visit = []
    for i in range(dim):
        can_visit.append([])
        for j in range(dim):
            if (A[i][j] != 0) :
                can_visit[i].append(j)
    return can_visit

def circle_finder(A, can_visit):
    dim = A.shape[0]

    for start in range(dim):
        path = [start]
        actual = start
        while (path != []):
            if (can_visit[actual] != []):
                # step to the next neighbour
                next_act = can_visit[actual][0]
                can_visit[actual].remove(next_act)
                actual = next_act
                path.append(actual)
                if (actual == start):
                    return (path)
            else:
                # backtrack
                path.remove(actual)
                if (path!=[]):
                    actual = path[-1]
    return ([])

def collapse_circle(A, circle, not_circle):
    "Returns a smaller matrix where the circle is represented by a new node"

    newdim = len(not_circle)+1

    # deleting rows that correspond to the circle
    collapsed = A
    collapsed = np.delete(collapsed, circle, 0)
    collapsed = np.delete(collapsed, circle, 1)

    # appending new row and column at the end for the new node
    collapsed = np.vstack((collapsed, np.zeros((1, newdim-1))))
    collapsed = np.hstack((collapsed, np.zeros((newdim, 1))))

    return collapsed


def calculate_mapping(not_circle):
    #calculates mapping
    shrink_map = {}
    expand_map = {}
    for i in range(len(not_circle)):
        shrink_map[not_circle[i]] = i
        expand_map[i] = not_circle[i]

    return shrink_map, expand_map


def fill_outgoing_weights(A, collapsed, circle, shrink_map):
    # gets the original matrix A, updates the collapsed matrix with outgoing arc weights, gets back the rec2 reconstruction array
    dim = A.shape[0]
    not_circle = np.arange(dim)
    not_circle = np.delete(not_circle, circle)

    A1 = A.copy()

    for i in not_circle:
        A1[i,:] = -np.inf
    for i in circle:
        A1[:,i] = 0

    max_ind = np.argmax(A1, 0)
    max_val = np.amax(A1, 0)


    for i in not_circle:
        collapsed[-1,shrink_map[i]] = max_val[i]
    reconstruct = max_ind

    return collapsed, reconstruct


def fill_incoming_weights(A, collapsed, circle, not_circle):
    #we cut the recurring double node from circle
    circle = circle[:-1]

    # construct matrix of circle entities
    circle_mtx = A[:, circle][circle, :]

    #calculating mappings
    shrink_map_nc, expand_map_nc = calculate_mapping(not_circle)

    maxspan_circle = []
    maxspan_values = np.zeros(len(circle))
    for i in range(len(circle)):
        maxspan_circle.append(Edmonds(circle_mtx, i))

    # calculate maxspan values from matrices
    for i in range(len(circle)):
        maxspan_values[i] = sum(sum(maxspan_circle[i]))

    mtx = np.zeros((len(not_circle), len(circle)))
    for i in range(len(not_circle)):
        for j in range(len(circle)):
            mtx[i, j] = maxspan_values[j] + A[expand_map_nc[i], circle[j]]

    # filling in the last column of the collapsed mtx
    collapsed[:-1, -1] = np.amax(mtx,1)

    # rec1 stores the path, connections(i)= j means that nc element i connects to circle element j, which is the root of the internal spanning tree
    rec1 = np.argmax(mtx, 1)


    return collapsed, rec1


r2 = np.array([0, 5, 5, 15])
r3 = np.array([20, 0, 5, 30])
r4 = np.array([10, 20, 0, 5])
r5 =np.array([5, 10, 15, 0])

A = np.array([r2, r3, r4, r5])


ran = np.random.rand(6,6)



root = [0, 15, 0, 0]

span_value = np.zeros(len(root))
for i in range(len(root)):
    span_value[i] = sum(sum(Edmonds(A, i)))

max_parse_values = span_value + root
root_ind = np.argmax(max_parse_values)

final_parse = Edmonds(A,root_ind)
final_parse[root_ind, root_ind] = root[root_ind]

print(final_parse)