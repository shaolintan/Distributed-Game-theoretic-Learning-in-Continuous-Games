import numpy as np
import networkx as nx

from operator import itemgetter

# initiate the nodes with given value
def add_dynamic_attributes(g,value):
    a={}
    i=0
    for d1 in g.nodes():
        a[d1]=value[i]
        i=i+1
    nx.set_node_attributes(g,'prob',a)        
    return g

def one_step_evol(G,gm):
    f={}
    for d1 in G.nodes_iter():
        f[d1]=0
        for d2 in G.neighbors_iter(d1):
            f[d1]=f[d1]+np.dot(np.dot([G.node[d1]['prob'],1-G.node[d1]['prob']],gm),
                             [G.node[d2]['prob'],1-G.node[d2]['prob']])
    s={}
    for d1 in G.nodes_iter():
        s[d1]=f[d1]
        for d2 in G.neighbors_iter(d1):
            s[d1]+=f[d2]
    for d1 in G.nodes_iter():
        for d2 in G.neighbors_iter(d1):
            G.node[d1]['prob']=G.node[d1]['prob']+f[d2]/float(s[d1])*(G.node[d2]['prob']
                            -G.node[d1]['prob'])
    return G

def evol(G,gm,dis,M):
    G=add_dynamic_attributes(G,dis)
    seq={}
    for d in G.nodes_iter():
        seq[d]=[]
    for i in range(M):
        for d in G.nodes_iter():
            seq[d].append(G.node[d]['prob'])
        G=one_step_evol(G,gm)
    return (G,seq)



def DB_one_step_evol(G,gm,w):
    f={}
    for d1 in G.nodes_iter():
        f[d1]=0
        for d2 in G.neighbors_iter(d1):
            f[d1]=f[d1]+np.dot(np.dot([G.node[d1]['prob'],1-G.node[d1]['prob']],gm),
                             [G.node[d2]['prob'],1-G.node[d2]['prob']])
        f[d1]=1-w+w*f[d1]
    s={}
    for d1 in G.nodes_iter():
        s[d1]=0
        for d2 in G.neighbors_iter(d1):
            s[d1]+=f[d2]
    m={}
    for d1 in G.nodes_iter():
        m[d1]=0
        for d2 in G.neighbors_iter(d1):
            m[d1]=m[d1]+f[d2]/float(s[d1])*G.node[d2]['prob']
    for d1 in G.nodes_iter():
        G.node[d1]['prob']=m[d1]
    return G

def DB_evol(G,gm,w,dis,M):
    G=add_dynamic_attributes(G,dis)
    seq={}
    for d in G.nodes_iter():
        seq[d]=[G.node[d]['prob']]
    for i in range(M):
        G=DB_one_step_evol(G,gm,w)
        for d in G.nodes_iter():
            seq[d].append(G.node[d]['prob'])
    return (G,seq)


# the one step updating in continuous imitation dynamics
def obtain_fitness(G,gm):
    f={}
    for d1 in G.nodes_iter():
        f[d1]=0
        for d2 in G.neighbors_iter(d1):
            f[d1]=f[d1]+np.dot(np.dot([G.node[d1]['prob'],1-G.node[d1]['prob']],gm),
                             [G.node[d2]['prob'],1-G.node[d2]['prob']])
    nx.set_node_attributes(G,'fitness',f)
    return G

def imi_one_step_evol(G,w)
    det={}
    for d1 in G.nodes_iter():
        det[d1]=0
        for d2 in G.neighbors_iter(d1):
            det[d1]=det[d1]+(np.tanh(w*(f[d2]-f[d1]))+1)*(G.node[d2]['prob']-G.node[d1]['prob'])/float(2*G.degree(d1))
    for d1 in G.nodes_iter():
        G.node[d1]['prob']+=det[d1]
    return G

def imi_evol(G,gm,dis,w,M):
    G=add_dynamic_attributes(G,dis)
    seq=[]
    seq2={}
    for i in range(M):
        s=0
        for d in G.nodes_iter():
            seq2[d].append(G.node[d]['prob'])
            s+=G.node[d]['prob']*G.degree(d)
        seq.append(s)
        G=obtain_fitness(G,gm)
        G=imi_one_step_evol(G,w)
    return (G,seq,seq2)

# intervention of the imitation dynamics
def award_the_best(G,u):
    f=nx.get_node_attributes(G,'fitness')
    f_1=sorted(f.iteritems(),key=itemgetter(1),reverse=True)
    f[f_1[0][0]]+=u
    nx.set_node_attritutes(G,'fitness',f)
    return G

def interven_imi_evol_1(G,gm,dis,w,M,u):
    G=add_dynamic_attributes(G,dis)
    seq=[]
    for i in range(M):
        s=0
        for d in G.nodes_iter():
            s+=G.node[d]['prob']*G.degree(d)
        seq.append(s)
        G=obtain_fitness(G,gm)
        G=award_the_best(G,u)
        G=imi_one_step_evol(G,w)
    return (G,seq)

def proportional_award(G,u):
    f=nx.get_node_attributes(G,'fitness') 
    s=0
    for d in G.nodes_iter():
        s+=f[d]
    for d in G.nodes_iter():
        f[d]+=f[d]/float(s)*u
    nx.set_node_attritutes(G,'fitness',f)
    return G
    
def interven_imi_evol_2(G,gm,dis,w,M,u):
    G=add_dynamic_attributes(G,dis)
    seq=[]
    for i in range(M):
        s=0
        for d in G.nodes_iter():
            s+=G.node[d]['prob']*G.degree(d)
        seq.append(s)
        G=obtain_fitness(G,gm)
        G=proportional_award(G,u)
        G=imi_one_step_evol(G,w)
    return (G,seq)   
    
    
                                                        
            
    
    
    
