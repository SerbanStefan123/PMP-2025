from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


smartHome=DiscreteBayesianNetwork([('O', 'H'), ('O', 'W'), ('H', 'E'),('H', 'R'),('W', 'R'),('R', 'C')])
pos = nx.circular_layout(smartHome)
nx.draw(smartHome, with_labels=True, pos=pos, alpha=0.5, node_size=2000)


CPD_O = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]])
   

CPD_H = TabularCPD(variable='H', variable_card=2,
values=[[0.9, 0.2], 
[0.1, 0.8,]],  
evidence=['O'],
evidence_card=[2])

CPD_W = TabularCPD(variable='W', variable_card=2,
values=[[0.1, 0.6], 
[0.9, 0.4,]],      
evidence=['O'],
evidence_card=[2])

CPD_E = TabularCPD(variable='E', variable_card=2,
values=[[0.8, 0.2], 
[0.2, 0.8,]],       
evidence=['H'],
evidence_card=[2])

CPD_C = TabularCPD(variable='C', variable_card=2,
values=[[0.85, 0.40], 
[0.15, 0.60]],        
evidence=['R'], 
evidence_card=[2])


CPD_R = TabularCPD(variable='R', variable_card=2,
values=[[0.6, 0.9, 0.3,0.5],  
[0.4, 0.1, 0.7 , 0.5]],     
evidence=['H', 'W'],
evidence_card=[2, 2])

smartHome.add_cpds(CPD_O,CPD_H,CPD_W,CPD_E,CPD_C,CPD_R)
assert smartHome.check_model()

pos = nx.circular_layout(smartHome)
nx.draw(smartHome, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

infer = VariableElimination(smartHome)

posterior_H = infer.query(['H'], evidence={'C': 0})
print(posterior_H)

posterior_E = infer.query(['E'], evidence={'C': 0})
print(posterior_E)

posterior_p = infer.query(["H","W"], evidence={"C": 0})
print(posterior_p)

