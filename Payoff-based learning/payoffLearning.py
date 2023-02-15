import networkx as nx
import numpy as np
import random as rd
import math as mt

# initial the position of each agent
def add_dynamic_attributes(g,a,b):
    p={}
    nodes=nx.nodes(g)
    for d in nodes:
        p[d]=np.array([rd.uniform(a,b),rd.uniform(a,b)])
    nx.set_node_attributes(g,'position',p)
    return g

# Generate an initial set of candidate players and its coordinate directions
def generate_players_and_directions(g):
    players={}
    for d in nx.nodes(g):
        players[d]=[np.array((0, 1)), np.array((0, -1)), np.array((1, 0)), np.array((-1, 0))]
    return players

# calculate the payoff of the given node
def calculate_payoff(g,node):
    payoff=0
    for d in g.neighbors(node):
        a=g.node[node]['position']-g.node[d]['position']
        payoff=payoff-np.linalg.norm(a)
    return payoff

# either return a successful position or reduced player set or shorter step     
def one_step_updating(g,players,step,rho):
    if players:
        node=rd.choice(players.keys())
        direct=players[node].pop()
        p1=0
        p2=0
        mid_position=g.node[node]['position']+step*direct
        for d in g.neighbors(node):
            a=g.node[node]['position']-g.node[d]['position']
            b=mid_position-g.node[d]['position']
            p1=p1-np.linalg.norm(a)
            p2=p2-np.linalg.norm(b)
        if p2-p1>0.001*step*step:
            g.node[node]['position']=mid_position
            players=generate_players_and_directions(g)
        else:
            if not players[node]:
                del players[node]
    else:
        step=rho*step
        players=generate_players_and_directions(g)
    return [players,step]

def generate_position(g):
    x_position={}
    y_position={}
    for d in g.nodes():
        x_position[d]=[]
        y_position[d]=[]
    position=(x_position,y_position)
    return position

# get the positions of each agents
def get_positions(g,position):
    for d in g.nodes():
        position[0][d].append(g.node[d]['position'][0])
        position[1][d].append(g.node[d]['position'][1])
    return position
            
# the learning dynamics
def payoff_based_learning(g,step_ini,step_fin,rho):
    position=generate_position(g)
    position=get_positions(g,position)
    players=generate_players_and_directions(g)
    step=step_ini
    while step>step_fin:
        mid=one_step_updating(g,players,step,rho)
        players=mid[0]
        step=mid[1]
        position=get_positions(g,position)
    return position
        
        
        
    
    


