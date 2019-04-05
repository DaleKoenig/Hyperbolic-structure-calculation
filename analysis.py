import spherogram
from spherogram import links
import numpy as np
import sympy
from scipy.optimize import least_squares
from functools import lru_cache
import ast

def reverse_edge(edge):
    return (edge[0].opposite(),edge[1].opposite())

def opposite_color(color):
    if color == 'white':
        return 'black'
    if color == 'black':
        return 'white'

def parse(code):
    "Given an PD code as a string, return a PD code as a list of tuples"
    return ast.literal_eval(code)

def get_adj_faces(face_i,faces):
    """
    Args:
        face_i -- index in faces of the face to find adjacent faces to
        faces -- list of all faces for the link
    """
    return [f for f in faces if any(reverse_edge(e) in faces[face_i] for e in f)]

def compute_k(edge):
    #MUST CHECK if edge goes left to right as viewed by region or right to left.
    #currently assumes left to right
    if(edge[0].strand_index in [1,3] and edge[1].strand_index in [0,2]):
        return -1
    elif(edge[0].strand_index in [0,2] and edge[1].strand_index in [1,3]):
        return 1
    else:
        return 0

def compute_kappa(edge_1,edge_2,oriented_edges):
    if edge_1 in oriented_edges and edge_2 in oriented_edges:
        return 1
    if edge_1 not in oriented_edges and edge_2 not in oriented_edges:
        return 1
    return -1

@lru_cache()
def get_f_n(shape_params,n):
    if n < 3:
        raise ValueError("parameter n must be at least 3")
    elif n == 3:
        return [1-s for s in shape_params] # this is assigning the first entry in shape_params to correspond to zeta_2 in the notation of the paper
    elif n == 4:
        return [1 - shape_params[i] - shape_params[(i+1) % len(shape_params)] for i in range(len(shape_params))]
    else:
        f_twoback = get_f_n(shape_params,n-2)
        f_oneback = get_f_n(shape_params,n-1)
        shape_n = shape_params[2:] + shape_params[:2]
        return [f_oneback[i] - shape_n[i]*f_twoback[i] for i in range(len(shape_params))]

def analyze(L):
    print('Analyzing link with pair code {}.'.format(L.PD_code()))
    # Collect edges.  Each edge is a (CrossingStrand, CrossingEntryPoint) pair, where the latter is the 'head' of the oriented edge.
    # Note that comparing a (CrossingStrand, CrossingEntryPoint) pair to a (CrossingStrand, CrossingStrand) pair (as are given in the faces) still works properly since CrossingEntryPoint inherits from CrossingStrand.
    oriented_edges = [(x.opposite(),x) for x in L.crossing_entries()]
    # every edge appears once as-is and once reversed in the faces
    faces = [[(x.opposite(),x) for x in face] for face in L.faces()] #hopefully right orientation
    # We create a local name for the set of crossings for consistency
    crossings = L.crossings

    # First color all the regions.  The colors are referenced by the list 'coloring' which stores the colors as strings "black" or "white"
    coloring = ['']*len(faces)
    coloring[0] = 'black'
    to_add = [faces.index(adj) for adj in get_adj_faces(0,faces)]
    for to_add_face in to_add:
        coloring[to_add_face] = 'white'
    while(to_add):
        face_i = to_add.pop()
        for nbr in get_adj_faces(face_i,faces):
            nbr_i = faces.index(nbr)
            if not coloring[nbr_i]:
                coloring[nbr_i] = opposite_color(coloring[face_i])
                to_add.append(nbr_i)

    # Next we iterate through all edges by iterating through black faces and assigning indices and names to the corresponding edge
    # At the same time, we assign indices and names to the reversed edge (bordering a white region)
    
    edge_indices = {} # used to index the edge in arrays or lists (such as 'edge_names' below)
    edge_names = [] # human readable names

    count = 0
    for face_i,face in enumerate(faces):
        if coloring[face_i] == 'white':
            continue
        for edge in face:
            edge_indices[edge] = count*2
            edge_indices[reverse_edge(edge)] = count*2+1
            edge_names.append('u' + str(count+1))
            edge_names.append('v' + str(count+1))
            count += 1

    labels = np.zeros(len(crossings) + 2*len(oriented_edges), dtype=np.complex_) #initial value estimates for all variables
    crossing_labels = labels[:len(crossings)] # reference crossing labels by slice
    edge_labels = labels[len(crossings):] # reference edge labels by slice
    edge_vars = sympy.symbols(edge_names)
    crossing_vars = sympy.symbols(['w' + str(k) for k in range(len(crossing_labels))])

    #labels = np.zeros(len(crossings+2*len(oriented_edges)), dtype=np.complex_)
    equations = []

    for crossing in L.crossings: # initial crossing label values
        if crossing.sign == 1:
            crossing_labels[crossing.label] = 0+.5j
        else:
            crossing_labels[crossing.label] = 0-.5j
    # Now lets go through each face and create the equations
    for face_i,face in enumerate(faces):
        if len(face) < 2:
            return
        if coloring[face_i] == 'black': # Add edge label equations (for the two sides of each edge) only if the color is black
            for edge in face:
                edge_labels[edge_indices[edge]] = -.5-.5j # initial edge label value for black side
                edge_labels[edge_indices[reverse_edge(edge)]] = .5-.5j #initial edge label value for white side
                k = compute_k(edge) #Set this to +-1 appropriately as per paper
                equations.append(edge_vars[edge_indices[edge]]-edge_vars[edge_indices[reverse_edge(edge)]]-k)
        if len(face) > 2: # generate equations for faces with more than 2 sides
            shape_params = []
            kl = compute_kappa(face[0],face[-1],oriented_edges)
            el = edge_vars[edge_indices[face[-1]]]
            e1 = edge_vars[edge_indices[face[0]]]
            w1 = crossing_vars[face[0][0][0].label]
            shape_par = kl*w1/el/e1
            shape_params.append(shape_par)
            for i in range(len(face) - 1):
                e_i = edge_vars[edge_indices[face[i]]]
                e_ii = edge_vars[edge_indices[face[i+1]]]
                w_i = crossing_vars[face[i+1][0][0].label]
                k_i = compute_kappa(face[i],face[i+1],oriented_edges)
                shape_par = k_i*w_i/e_i/e_ii
                shape_params.append(shape_par)
            edge_prod = 1
            for e in face:
                edge_prod *= edge_vars[edge_indices[e]]
            f_n = get_f_n(tuple(shape_params),len(shape_params)) # get list of f_n functions
            equations = equations + f_n[:3]
    for face_i,face in enumerate(faces): # special initial value assignments for bigons
        if len(face) == 2:
            for edge in face:
                equations.append(edge_vars[edge_indices[edge]]) # enforce edge_label == 0 for sides of 2 sided regions
                edge_labels[edge_indices[edge]] = 0 # Reset edge label to 0
    print("We run Newton-Raphson method to solve the following system of equations:")
    for eq in equations:
        print("{} = 0".format(eq))
    f_temp = sympy.lambdify(crossing_vars+edge_vars,equations,'numpy')
    Jac = sympy.Matrix([[sympy.diff(eq,var) for var in crossing_vars + edge_vars] for eq in equations])
    Df_temp = sympy.lambdify(crossing_vars + edge_vars, Jac,'numpy')
    f = lambda x : f_temp(*x)
    Df = lambda x: Df_temp(*x)
    newton_iter = 30
    for _ in range(newton_iter):
        f_x = f(labels)
        Df_x = Df(labels)
        q,r = np.linalg.qr(Df_x)
        incx = np.linalg.solve(r,-np.matmul(q.conj().T,f_x))
        labels = np.around(labels+incx,6)
    #output time
    print("\nThe face with vertices {} is black.  Other faces can be colored in a checkerboard fashion.\n'u' labels correspond to the black sides of the edges, and 'v' labels to the white sides.".format([edge[1][0] for edge in faces[0]]))
    for i in range(len(crossings)):
        print("w{} = {}".format(i,labels[i]))
    count = 0
    print('\nCalculated values:')
    for face_i,face in enumerate(faces):
        if coloring[face_i] == 'white':
            continue
        for edge in face:
            print('Edge from vertex {} to vertex {}'.format(edge[0][0],edge[1][0]))
            print('u{} = {}'.format(str(count+1),labels[len(crossings)+count*2]))
            print('v{} = {}'.format(str(count+1),labels[len(crossings)+count*2+1]))
            count += 1

if __name__ == '__main__':
    print("Input a link by PD (Planar Diagram) code.  See, for example, https://arxiv.org/abs/1309.3288.")
    print("Note, with this method it seems like spherogram may not accept signs in the input, so all links will be alternating.")
    print("Vertex and edge labels should start at 0.")
    print("Example input: {}".format('(1, 7, 2, 6), (5, 3, 6, 2), (7, 4, 0, 5), (3, 0, 4, 1)'))
    print("So this know has 4 vertices labelled 0, 1, 2, and 3")
    print("We can draw the knot starting with the edge labelled 0, which connects vertex 2 and 3")
    print("The path from vertex 3 to vertex 0 with label 1, then on to vertex 1 with label 2, then to vertex 3 with label 3, etc.")
    print("We can see that vertex 0 will end up surrounded by edges labelled 1, 7, 2, an 6, precisely the first tuple given.")
    print("\nInput a PD code:")
    inp = parse(input())
    #L = links.Link('L8a9')
    L = links.Link(inp)
    analyze(L)
    
    
"""#matrix way of calculating face equation
            kl = compute_kappa(face[0],face[-1],oriented_edges)
            el = edge_vars[edge_indices[face[-1]]]
            e1 = edge_vars[edge_indices[face[0]]]
            w1 = crossing_vars[face[0][0][0].label]
            m = sympy.Matrix([[0,-kl*w1],[el*e1,-el*e1]])
            #print(m)
            for i in range(len(face) - 1):
                e_i = edge_vars[edge_indices[face[i]]]
                e_ii = edge_vars[edge_indices[face[i+1]]]
                w_i = crossing_vars[face[i+1][0][0].label]
                k_i = compute_kappa(face[i],face[i+1],oriented_edges)
                m_prod = sympy.Matrix([[0,-k_i*w_i],[e_i*e_ii,-e_i*e_ii]])
                m = m_prod * m
            equations.append(m[0,0] - m[1,1])
            equations.append(m[0,1])
            equations.append(m[1,0]) #This is the default way of calculating, and doesn't converge to the right thing.  Need to try recursive formula"""

"""def turks_head_test():
    all_vars = sympy.symbols('w1, w2, w3, w4, w5, w6, w7, w8, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, \
v14, v15, v16, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16')
    w1, w2, w3, w4, w5, w6, w7, w8, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, \
v14, v15, v16, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16 = all_vars
    equations = [u1 - v1 - 1, u2 - v2 - 1, u3 - v3 - 1, u4 - v4 - 1, u5 - v5 - 1, u6 - v6 - 1, u7 - v7 - 1, u8 - v8 - 1, u9 - v9 - 1, 
    u10 - v10 - 1, u11 - v11 - 1, u12 - v12 - 1, u13 - v13 - 1, u14 - v14 - 1, u15 - v15 - 1, u16 - v16 - 1, 
    w1*u4 + w3*u12 - u16*u4*u12, w3*u8 + w5*u16 - u16*u4*u8, w5*u12 + w7*u4 - u4*u8*u12, w2*v10 + w8*v2 - v14*v10*v2, w8*v6 + w6*v14 - v14*v10*v6, w6*v2 + w4*v10 - v10*v6*v2, 
    w1 + v1*v12, w7 + v12*v7, w4 - v1*v7, w1 - u1*u11, w4 + u1*u6, w6 + u6*u11, w1 + v11*v16, w3 + v5*v16, w6 - v11*v5, w5 - u3*u9, w2 + u3*u14, 
    w8 + u9*u14, w7 + v13*v8, w5 + v3*v8, w2 - v3*v13, w7 - u7*u13, w2 + u2*u13, w4 + u2*u7, w3 + v4*v15, w5 + v9*v4, w8 - v15*v9, w3 - u5*u15, w6 + u5*u10, w8 + u10*u15]

    
    Jac = sympy.Matrix([[sympy.diff(eq,var) for var in all_vars] for eq in equations])
    labels = np.zeros(len(all_vars), dtype=np.complex_)
    for i in range(0,8,2):
        labels[i] = .5j
    for i in range(1,8,2):
        labels[i] = -.5j
    for i in range(8,24):
        labels[i] = -.5+.5j
    for i in range(24,40):
        labels[i] = .5+.5j
    f_temp = sympy.lambdify(all_vars,equations,'numpy')
    Df_temp = sympy.lambdify(all_vars,Jac,'numpy')
    f = lambda x : f_temp(*x)
    Df = lambda x: Df_temp(*x)
    for _ in range(1):
        f_x = f(labels)
        Df_x = Df(labels)
        q,r = np.linalg.qr(Df_x)
        incx = np.linalg.solve(r,-np.matmul(q.conj().T,f_x))
        labels = np.around(labels + incx,6)
    print(labels)"""