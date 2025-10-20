from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('S','O'),('S','L'),('S','M'),('L','M')])

cpd_S = TabularCPD('S',2,[[0.6],[0.4]])
cpd_O = TabularCPD('O',2,[[0.9,0.3],[0.1,0.7]],evidence=['S'],evidence_card=[2])
cpd_L = TabularCPD('L',2,[[0.7,0.2],[0.3,0.8]],evidence=['S'],evidence_card=[2])
cpd_M = TabularCPD('M',2,[[0.8,0.4,0.1,0.5],[0.2,0.6,0.9,0.5]],evidence=['S','L'],evidence_card=[2,2])

model.add_cpds(cpd_S,cpd_O,cpd_L,cpd_M)
print(model.check_model())

inference = VariableElimination(model)
print(inference.query(variables=['S'],evidence={'O':1,'L':1,'M':1}))
print(inference.query(variables=['S'],evidence={'O':0,'L':0,'M':0}))
