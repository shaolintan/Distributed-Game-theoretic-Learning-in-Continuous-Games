import networkx as nx
import sign_graph as sg
import numpy as np
import random as rd

g=nx.read_weighted_edgelist('EGFR_symm.txt')
#g=nx.Graph()
#g.add_edge(1,2,weight=-1)
#g.add_edge(2,3,weight=1)
#g.add_edge(3,4,weight=-1)
#g.add_edge(4,1,weight=1)

#g=sg.add_dynamic_attributes_2(g)
#g.node[1]['strategy']=-1
#g.node[2]['strategy']=-1
#g.node[3]['strategy']=1
#g.node[4]['strategy']=1

#rslt=sg.departe_graph(g)

#rslt2=sg.one_step_updating_2(g,rslt[0],rslt[1],rslt[2],rslt[4])

rslt=sg.evo_alg_4(g,100,0.9,1000,60)
#g1=sg.evo_alg_2(g,3000,10)
#rslt2=sg.partition(g1)
# g2=sg.evo_alg(g,0.02,2000,5)
# rslt2=sg.balanced_level(g2)

