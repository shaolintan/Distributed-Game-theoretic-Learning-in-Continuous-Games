import multiagent_dynamics2 as md
import networkx as nx
import ban1

def dynamics(g,T):
    i=0
    while i<T/10:
        [g,k]=md.update_payoff(g)
        [g,b]=md.update_strategy1(g)
        if b==0: break
        i+=1
    while i>=T/10 and i<T:
        [g,k]=md.update_payoff(g)
        [g,b]=md.update_strategy2(g)
        if b==0: break
        i+=1
    return g

g=nx.erdos_renyi_graph(15,0.2)
g=md.add_dynamic_edge_attribute(g)
g=md.add_dynamic_attributes(g)
[g,total1]=md.update_payoff(g)
g1=dynamics(g,500000)
[g1,total2]=md.update_payoff(g)
print (2*g.size()-total1)/4
print (2*g.size()-total2)/4





