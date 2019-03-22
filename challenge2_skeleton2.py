# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter


def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 

    The assumption is that two connected nodes are
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors

    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes
    attr : dict
       A dict of attributes, either location, employer or college attributes.
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict
       A dict of attributes, either location, employer or college attributes.
       key is a node (from empty), value is a list of attribute values. Here
       only 1 value in the list.
     """
    predicted_values = {}
    for n in empty:
        nbrs_attr_values = []
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n] = []
        if nbrs_attr_values:  # non empty list
            # count the number of occurrence each value and returns a dict
            cpt = Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a, nb_occurrence = max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values


def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results.

    Parameters
    ----------
    groundtruth :  : dict
       A dict of attributes, either location, employer or college attributes.
       key is a node, value is a list of attribute values.
    pred : dict
       A dict of attributes, either location, employer or college attributes.
       key is a node, value is a list of attribute values.

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction = 0
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction += 1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])
            # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction * 100 / sum(len(v) for v in pred.values())


# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")
print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
college = {}
location = {}
employer = {}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)

print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes = []
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("Your mission, find attributes to %d users with empty profile" % len(empty_nodes))

# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'
employer_predictions = naive_method(G, empty_nodes, employer)
groundtruth_employer = {}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
empty_nodes = []
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
with open('mediumLocation.pickle', 'rb') as handle:
    groundtruth_location = pickle.load(handle)
with open('mediumCollege.pickle', 'rb') as handle:
    groundtruth_college = pickle.load(handle)
"""
result=evaluation_accuracy(groundtruth_employer,employer_predictions)
print("%f%% of the predictions are true" % result)
print("Very poor result!!! Try to do better!!!!")
"""


# -------------- Statistics -------------------------------
def properties(g):
    """
    Computes simple and classic graph metrics.

    Parameters
    ----------
    g : graph
       A networkx graph
    """
    # networkx short summary of information for the graph g
    print(nx.info(g))

    # Draw the degree distribution. Powerlow distribution for a real (complex) network
    plt.figure(num=None)
    fig = plt.figure(1)
    degree_sequence = [d for n, d in g.degree()]  # degree sequence
    print("Degree sequence %s" % degree_sequence)
    plt.hist(degree_sequence, bins='auto')
    plt.title("powerlaw degree distribution")
    plt.ylabel("# nodes")
    plt.xlabel("degree")
    plt.show()
    pylab.close()
    del fig

    precomputed_eccentricity = nx.eccentricity(g)  # costly step, we save time here!
    print("Graph density %f" % nx.density(g))
    print("Diameter (maximum eccentricity): %d" % nx.diameter(g, precomputed_eccentricity))
    print("Radius (minimum eccentricity): %d" % nx.radius(g,
                                                          precomputed_eccentricity))  # The radius is the minimum eccentricity.
    print("Mean eccentricity (eccentricity(v) = the maximum distance from v to all other nodes): %s" % np.mean(
        list(precomputed_eccentricity.values())))
    print("Center is composed of %d nodes (nodes with eccentricity equal to radius)" % len(
        nx.center(g, precomputed_eccentricity)))
    print("Periphery is composed of %d nodes (nodes with eccentricity equal to the diameter)" % len(
        nx.periphery(g, precomputed_eccentricity)))
    print("Mean clustering coefficient %f" % np.mean(list(nx.clustering(g).values())))
    total_triangles = sum(nx.triangles(g).values()) / 3
    print("Total number of triangles in graph: %d" % total_triangles)


properties(G)


# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes


# and compare with the ground truth (what you should have predicted)
# user precision and recall measures
def Corr(college, graph):
    N = len(college)
    card = 0
    for node in college:  # pour chaque noeud
        similar = False
        for neigh in graph.neighbors(node):  # pour chacun de ses voisins
            if neigh in college:
                for col in college[neigh]:
                    if col in college[node]:  # si ils ont un emploi en commun...
                        similar = True
        if similar:
            card = card + 1
    return card / N


def Corr2(college, graph):
    N = 0
    card = 0
    for node in college:  # pour chaque noeud
        similar = False
        counted = False
        for neigh in graph.neighbors(node):  # pour chacun de ses voisins
            if neigh in college:
                counted = True
                for col in college[neigh]:
                    if col in college[node]:  # si ils ont un emploi en commun...
                        similar = True

        if similar:
            card = card + 1
        if counted:
            N += 1
    return card / N


def empty_neigh(graph, empties):
    N = len(empties)
    card = 0
    for node in empties:
        b = False
        for neigh in graph.neighbors(node):
            if neigh in empties:
                b = True
        card = card + b
    return card / (2 * N)


locations = []
for node in location:
    for l in location[node]:
        locations.append(l)
locations = dict(Counter(locations))
employers = []
for node in employer:
    for l in employer[node]:
        employers.append(l)
employers = dict(Counter(employers))
colleges = []
for node in college:
    for l in college[node]:
        colleges.append(l)
colleges = dict(Counter(colleges))
labeled = []
for node in G.nodes:
    if node not in empty_nodes:
        labeled.append(node)

empty_node_neighs = {}
for node in empty_nodes:
    tot = 0
    sc = 0
    for neigh in G.neighbors(node):
        tot += 1
        if neigh in empty_nodes:
            sc += 1
    empty_node_neighs[node] = sc / tot


def acc_from(groundtruthl, l):
    tot = 0
    good = 0
    for item in l:
        if item in groundtruthl:
            tot += 1
            isThere = False
            for attr in l[item]:
                if attr in groundtruthl[item]:
                    isThere = True
            if isThere:
                good += 1
    return good / tot


# preprocessing : if an empty node has only one neighbor, and this neighbor isn't an empty node, he has the same attributes

print("---- STATS ----\n")


def remove_periphery(G, empty_nodes, labeled, location, employer, college):
    newlabeled = labeled
    newempty = empty_nodes
    pred_location, pred_employer, pred_college=location, employer, college
    for node in empty_nodes:
        if len(list(G.neighbors(node))) == 1:
            n = G.neighbors(node)
            if n not in empty_nodes:
                if n in location:
                    pred_location[node] = pred_location[n]
                if n in employer:
                    pred_employer[node] = pred_employer[n]
                if n in college:
                    pred_college[node] = pred_college[n]
            newlabeled.append(node)
            newempty.remove(node)
    return pred_location, pred_employer, pred_college, newempty, newlabeled


# STEP 2 : empty nodes with only labeled neighbors
"""
nb_takenoff=0
for node in empty_nodes:
    if empty_node_neighs[node] == 0 :
        nb_takenoff +=1
        neigh_attr_values=[]
        for neigh in G.neighbors(node):
            if neigh in employer :
                for val in employer[neigh]:
                    neigh_attr_values.append(val)
        if neigh_attr_values :
            cpt=Counter(neigh_attr_values)
            a,nb_occ=max(cpt.items(), key=lambda t: t[1])
            pred_employer[node]=a
print(acc_from(groundtruth_employer,pred_employer))
"""


def iterate(G, nb_neigh, empty_nodes, location, employer, college):
    empties = {}
    for node in G:
        nb = 0
        for neigh in G.neighbors(node):
            if neigh in empty_nodes:
                nb += 1
        empties[node] = nb
    for node in empty_nodes:
        if empties[node] == nb_neigh:
            nbr_loc = []
            nbr_emp = []
            nbr_col = []
            for n in G.neighbors(node):
                if n not in empty_nodes:
                    if n in location:
                        for l in location[n]:
                            nbr_loc.append(l)
                    if n in employer:
                        for e in employer[n]:
                            nbr_emp.append(e)
                    if n in college:
                        for c in college[n]:
                            nbr_loc.append(c)
            action = False
            if nbr_loc:
                nbr_loc = Counter(nbr_loc)
                a, nb = max(nbr_loc.items(), key=lambda t: t[1])
                location[node] = a
                action = True
            if nbr_emp:
                nbr_emp = Counter(nbr_emp)
                a, nb = max(nbr_loc.items(), key=lambda t: t[1])
                employer[node] = a
                action = True
            if nbr_col:
                nbr_col = Counter(nbr_col)
                a, nb = max(nbr_col.items(), key=lambda t: t[1])
                college[node] = a
                action = True
            if action:
                empty_nodes.remove(node)
    return location, employer, college, empty_nodes


def methode1():
    p_l, p_e, p_c, newempty, newlabeled = remove_periphery(G, empty_nodes, labeled, location, employer, college)
    empties = {}
    for node in G:
        nb = 0
        for neigh in G.neighbors(node):
            if neigh in newempty:
                nb += 1
        empties[node] = nb
    while newempty:
        for i in range(0, max(empties.values())):
            for j in range(0, 10):
                p_l, p_e, p_c, newempty = iterate(G, i, newempty, p_l, p_e, p_c)
    return p_l, p_e, p_c, newempty


def LocMatrix(attribut, lis):
    import numpy as np
    locationsList = attribut
    loca=dict(Counter[lis])
    LocM = np.zeros((len(locationsList), len(locationsList)))
    ind = dict(zip(locationsList, [i for i in range(len(locationsList))]))
    inv = dict(zip([i for i in range(len(locationsList))], locationsList))
    for loc1 in locationsList:
        for loc2 in locationsList:
            n2 = loca[loc1]
            n1 = 0
            for node in lis:
                if loc1 in lis[node]:
                    for neigh in G.neighbors(node):
                        if neigh in lis:
                            if loc2 in lis[neigh]:
                                n1 += 1
            LocM[ind[loc2]][ind[loc1]] = n1 / n2
    return LocM, ind, inv


def methode2():
    with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
        college = pickle.load(handle)
    with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
        location = pickle.load(handle)
    with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
        employer = pickle.load(handle)
    with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
        empty_nodes = pickle.load(handle)
    # p_l, p_e, p_c, newempty, newlabeled = remove_periphery(G, empty_nodes, labeled, location, employer, college)
    locations = []
    for node in location:
        for l in location[node]:
            locations.append(l)
    locations = dict(Counter(locations))
    employers = []
    for node in employer:
        for l in employer[node]:
            employers.append(l)
    employers = dict(Counter(employers))
    colleges = []
    for node in college:
        for l in college[node]:
            colleges.append(l)
    colleges = dict(Counter(colleges))
    p_l, p_e, p_c, newempty = location, employer, college, empty_nodes
    LocM, indL, invL = LocMatrix(locations, location)
    EmpM, indE, invE = LocMatrix(employers, employer)
    ColM, indC, invC = LocMatrix(colleges, college)
    while newempty:
        for node in newempty:
            neigh_locs = []
            neigh_cols = []
            neigh_emps = []
            for neigh in G.neighbors(node):
                if neigh in location:
                    for l in location[neigh]:
                        neigh_locs.append(l)
                if neigh in college:
                    for c in college[neigh]:
                        neigh_cols.append(c)
                if neigh in employer:
                    for e in employer[neigh]:
                        if e != 'athens university of economics and business':
                            neigh_emps.append(e)
            scores = []
            if neigh_locs:
                for l in neigh_locs:
                    i = np.argmax(LocM[indL[l]])
                    lo = invL[i]
                    element = (lo, LocM[indL[l]][np.argmax(LocM[indL[l]])])
                    scores.append(element)
                best_l, max_pr = scores[0]
                for l, pr in scores:
                    if pr > max_pr:
                        max_pr = pr
                        best_l = l
                p_l[node] = [best_l]
                newempty.remove(node)
            scores = []
            if neigh_cols:
                for c in neigh_cols:
                    i = np.argmax(ColM[indC[c]])
                    lo = invC[i]
                    element = (lo, ColM[indC[c]][np.argmax(ColM[indC[c]])])
                    scores.append(element)
                best_l, max_pr = scores[0]
                for l, pr in scores:
                    if pr > max_pr:
                        max_pr = pr
                        best_l = l
                p_c[node] = [best_l]
            if neigh_emps:
                for e in neigh_emps:
                    i = np.argmax(EmpM[indE[e]])
                    lo = invE[i]
                    element = (lo, EmpM[indE[e]][np.argmax(EmpM[indE[e]])])
                    scores.append(element)
                best_l, max_pr = scores[0]
                for l, pr in scores:
                    if pr > max_pr:
                        max_pr = pr
                        best_l = l
                p_e[node] = [best_l]
    return p_l, p_e, p_c, newempty


# p_l2, p_e2, p_c2, newempty=methode2()

def getallpossible(attribute):
    attr = []
    for node in attribute:
        for a in attribute[node]:
            if a not in attr:
                attr.append(a)
    return attr


def f_init(G, empty_nodes, location, employer, college):
    fVectors = {}
    locations = getallpossible(location)
    colleges = getallpossible(college)
    employers = getallpossible(employer)
    totLen = len(locations) + len(colleges) + len(employers)
    indC = dict(zip(colleges, [i for i in range(0, len(colleges))]))
    invC = dict(zip([i for i in range(0, len(colleges))], colleges))
    indE = dict(zip(employers, [i for i in range(len(colleges), len(colleges) + len(employers))]))
    invE = dict(zip([i for i in range(len(colleges), len(colleges) + len(employers))], employers))
    indL = dict(zip(locations, [i for i in range(len(colleges) + len(employers), totLen)]))
    invL = dict(zip([i for i in range(len(colleges) + len(employers), totLen)], locations))
    ind = {**indC, **indE, **indL}
    inv = {**invC, **invE, **invL}
    omega1 = [0 for i in range(len(colleges))]
    omega1.extend([1 for i in range(len(employers))])
    omega1.extend([1 for i in range(len(locations))])
    omega2 = [1 for i in range(len(colleges))]
    omega2.extend([0 for i in range(len(employers))])
    omega2.extend([1 for i in range(len(locations))])
    for node in G.nodes:
        if node not in empty_nodes:
            fVectors[node] = np.zeros((1, totLen)).tolist()[0]
            if node in location:
                for loc in location[node]:
                    fVectors[node][ind[loc]] = 1
            if node in college:
                for col in college[node]:
                    fVectors[node][ind[col]] = 1
            if node in employer:
                for emp in employer[node]:
                    fVectors[node][ind[emp]] = 1
        else:
            fVectors[node] = [0.5 for i in range(totLen)]
    return fVectors, omega1, omega2, ind, inv, locations, colleges, employers


def x_init(G):
    from random import randint
    xValues = {}
    for node in G.nodes:
        if node not in xValues:
            xValues[node] = {}
        for n in G.neighbors(node):
            x = randint(1, 2)
            if n not in xValues:
                xValues[n] = {}
                xValues[node].update({n: x})
                xValues[n].update({node: x})
            else:
                if node not in xValues[n]:
                    xValues[n].update({node: x})
                    xValues[node].update({n: x})
    return xValues


def predict(empty_nodes, location, employer, college, fVectors, colleges, employers, inv):
    pred_l, pred_c, pred_e = {}, {}, {}
    for node in empty_nodes:
        fVec = fVectors[node]
        colVect = fVec[:len(colleges)]
        empVect = fVec[len(colleges):len(colleges) + len(employers)]
        locVect = fVec[len(colleges) + len(employers):]
        bestColleges = np.argwhere(colVect == np.amax(colVect)).flatten().tolist()
        bestEmployers = np.argwhere(empVect == np.amax(empVect)).flatten().tolist()
        bestLocations = np.argwhere(locVect == np.amax(locVect)).flatten().tolist()
        if len(bestColleges) < 2:
            pred_c[node] = []
            for col in bestColleges:
                pred_c[node].append(inv[col])
        if len(bestEmployers) < 5:
            pred_e[node] = []
            for emp in bestEmployers:
                pred_e[node].append(inv[emp + len(colleges)])
        if len(bestLocations) < 2:
            pred_l[node] = []
            for loc in bestLocations:
                pred_l[node].append(inv[loc + len(colleges) + len(employers)])
    return pred_l, pred_e, pred_c


def coprofiling(G, empty_nodes, location, employer, college):
    location2, employer2, college2, newempty= location, employer, college, empty_nodes
    fVectors, omega1, omega2, ind, inv, locations, colleges, employers = f_init(G, newempty, location2, employer2,
                                                                                college2)
    xValues = x_init(G)
    nbnode = 0
    for xm in range (3) :
        for empty_node in newempty:
            print("Ongoing : %d / %d" % (nbnode, len(newempty)))
            fVectorsnext = fVectors
            circle1 = []
            circle2 = []
            for num in range(30):
                # print("Iteration %d sur 10"%num)
                # UPDATE f
                for neigh in G.neighbors(empty_node):
                    if xValues[neigh][empty_node] == 1:
                        circle1.append(neigh)
                    else:
                        circle2.append(neigh)
                for neigh in circle1:
                    if neigh in newempty:
                        L = len(omega1)
                        for i in range(L):
                            if omega1[i] == 1:
                                fVectorsnext[neigh][i] = fVectors[empty_node][i]
                                for n2 in circle1:
                                    if n2 != neigh:
                                        fVectorsnext[neigh][i] += fVectors[n2][i]
                                fVectorsnext[neigh][i] /= 1 + len(circle1)
                for neigh in circle2:
                    if neigh in newempty:
                        L = len(omega2)
                        for i in range(L):
                            if omega2[i] == 1:
                                fVectorsnext[neigh][i] = fVectors[empty_node][i]
                                for n2 in circle2:
                                    if n2 != neigh:
                                        fVectorsnext[neigh][i] += fVectors[n2][i]
                                fVectorsnext[neigh][i] /= 1 + len(circle2)
                for i in range(L):
                    den = 1
                    if omega1[i] == 1:
                        den += len(omega1)
                        for neigh in circle1:
                            fVectorsnext[empty_node][i] += fVectorsnext[neigh][i]
                    if omega2[i] == 1:
                        den += len(omega2)
                        for neigh in circle2:
                            fVectorsnext[empty_node][i] += fVectorsnext[neigh][i]
                    fVectorsnext[empty_node][i] /= den
                fVectors = fVectorsnext
                # UPDATE X
                xValuesnext = xValues
                for neigh in G.neighbors(empty_node):
                    soust = [fi - f0 for (fi, f0) in zip(fVectors[neigh], fVectors[empty_node])]
                    a = -0.7 * ((np.dot(omega1, soust)) ** 2)
                    b = -0.7 * ((np.dot(omega2, soust)) ** 2)
                    for nj in circle1:
                        soust = [fi - fj for (fi, fj) in zip(fVectors[neigh], fVectors[nj])]
                        a += 1 - 0.7 * ((np.dot(omega1, soust)) ** 2)
                    for nj in circle2:
                        soust = [fi - fj for (fi, fj) in zip(fVectors[neigh], fVectors[nj])]
                        b += 1 - 0.7 * ((np.dot(omega2, soust)) ** 2)
                    if neigh not in empty_nodes:
                        a += -7 * ((np.dot(omega1, fVectors[neigh]) - 1) ** 2)
                        b += -7 * ((np.dot(omega2, fVectors[neigh]) - 1) ** 2)
                    if a > b:
                        xValuesnext[neigh][empty_node] = 1
                        xValuesnext[empty_node][neigh] = 1
                    else:
                        xValuesnext[neigh][empty_node] = 2
                        xValuesnext[empty_node][neigh] = 2
                xValues = xValuesnext
            nbnode += 1
            predLoc, predEmp, predCol = predict(empty_nodes, location, employer, college, fVectors, colleges, employers,
                                                inv)
            if predLoc :
                print("Location (",len(predLoc.items()),") : ", evaluation_accuracy(groundtruth_location, predLoc))
            if predEmp :
                print("Employers (",len(predEmp.items()),") : ", evaluation_accuracy(groundtruth_employer, predEmp))
            if predCol :
                print("Colleges (",len(predCol.items()),") : ", evaluation_accuracy(groundtruth_college, predCol))
    return xValues, fVectors, predLoc, predEmp, predCol


with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)

fVectors, omega1, omega2, ind, inv, locations, colleges, employers = f_init(G, empty_nodes, location, employer, college)
xValues = x_init(G)
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
xValues, fVectors, predLoc, predEmp, predCol = coprofiling(G, empty_nodes, location, employer, college)


def neigh_empty_nb(G, empty_nodes):
    neigh_empty = {}
    for node in G.nodes:
        cnt = 0
        for neigh in G.neighbors(node):
            if neigh in empty_nodes:
                cnt += 1
        neigh_empty[node] = cnt
    return neigh_empty


def acc_repartition(predicted_attribute, real_attribute, neigh_empty, empty_nodes):
    repartition = {}
    for node in empty_nodes:
        if node in predicted_attribute:
            if neigh_empty[node] not in repartition:
                if node in real_attribute:
                    isThere = False
                    for attr in predicted_attribute[node]:
                        if attr in real_attribute[node]:
                            isThere = True
                    if isThere:
                        repartition[neigh_empty[node]] = (1, 1)  # (nb_true,nb_tot)
                    else:
                        repartition[neigh_empty[node]] = (0, 1)
            else:
                if node in real_attribute:
                    isThere = False
                    for attr in predicted_attribute[node]:
                        if attr in real_attribute[node]:
                            isThere = True
                    if isThere:
                        nb_true, nb_tot = repartition[neigh_empty[node]]
                        repartition[neigh_empty[node]] = (nb_true + 1, nb_tot + 1)  # (nb_true,nb_tot)
                    else:
                        nb_true, nb_tot = repartition[neigh_empty[node]]
                        repartition[neigh_empty[node]] = (nb_true, nb_tot + 1)
    return repartition


def shape_repartition(repartition):
    nb_empty_neighs = []
    nb_acc = []
    for key, value in sorted(repartition.items()):
        nb_empty_neighs.append(key)
        nb_acc.append(value[0] / value[1])
    return nb_empty_neighs, nb_acc

def influencers(G, location, centralities):
    targetLoc=['san francisco bay area']
    centr=[]
    for node, c in  sorted(centralities.items(), key=lambda kv:kv[1]) :
        if location[node]==targetLoc :
            centr.append((node,c))
    centr.reverse()
    return centr

def find_infl(G,location):
    SFpeople=[v for v in location if location[v]==['san francisco bay area']]
    SFremaining=list(SFpeople)
    ppldict={}
    for person in SFpeople :
        nb=0
        for neigh in G.neighbors(person):
            if neigh in SFpeople :
                nb+=1
            for nei in G.neighbors(neigh):
                if nei in SFpeople :
                    nb+=1
        ppldict[person]=nb
    ppl

def influenceurs(G,location):
    restau_location=['san francisco bay area']
    list_influenceurs=[]
    for node in G.nodes():
#        if node in location.keys()and location[node]==restau_location:
        k=0
        for neig in G.neighbors(node):
            if neig in location.keys():
                if location[neig]==restau_location:
                    k+=1
        if k>0:
            list_influenceurs.append((node,k))
    list_influenceurs=sorted(list_influenceurs,key=lambda x: x[1],reverse=True)
    return(list_influenceurs[:5])

def get_influencer(dic):
    dictx=list()
    for node in dic:
        dictx.append((node,len(dic[node])))
    dictx=sorted(dictx, key=lambda x: x[1], reverse=True)
    return(dictx[0][0])
def update(dic,liste):
    for ele in liste:
        for node in dic:
            if ele in dic[node]:
                dic[node].pop(dic[node].index(ele))
    return(dic)
def find_infl(G,location):
    restau_location=['san francisco bay area']
    influencers={}
    list_influencers=[]
    for node in G.nodes():
        liste=list()
        for nb in G.neighbors(node):
            if nb in location.keys():
                if location[nb]==restau_location:
                    liste.append(nb)
        influencers[node]=liste
    while len(list_influencers)<5:
        if len(influencers.keys())>0:
            best_influencer=get_influencer(influencers)
            list_influencers.append(best_influencer)
            a=influencers[best_influencer]
            del influencers[best_influencer]
            influencers=update(influencers,a)
    return(list_influencers)