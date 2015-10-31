""" Comverted from matlab code
Source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_1_sum_product/


 main_big_alarm_net

 This file creates the graph structure needed for the big alarm network.
 Creating the graph structure includes creating all nodes, placing the
 nodes into the stucture with the appropriate neighbors and assigning
 probability tables to the factor nodes. At the very end of this main
 file is a for loop which is documented. That for loop executes the
 inference algorithm accross the graph. At initialization, every variable
 node in the graph is unobserved. If you wish to do inference with some
 of the nodes observed you will need to indicate the observed variables
 before the inference procedure is executed."""

import numpy as np
from classes import Factor, VariableNode, FactorNode

f = np.array([ 0.2, 0.8]) 
fact = Factor(f) 
 
vn_Hypovolemia = VariableNode('vn:Hypovolemia', 2) 
fn_Hypovolemia = FactorNode('fn:Hypovolemia', fact) 
 
vn_Hypovolemia.addNode(fn_Hypovolemia) 
fn_Hypovolemia.addNode(vn_Hypovolemia) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [0.05, 0.95] 
fact = Factor(f) 
 
vn_LVFailure = VariableNode('vn:LVFailure', 2) 
fn_LVFailure = FactorNode('fn:LVFailure', fact)
 
vn_LVFailure.addNode(fn_LVFailure)
fn_LVFailure.addNode(vn_LVFailure)

# now we do the next node 
 
f = np.zeros((3, 2, 2))
f[:, 0, 0] = [ 0.95, 0.04, 0.01] 
f[:, 0, 1] = [ 0.98, 0.01, 0.01] 
f[:, 1, 0] = [ 0.01, 0.09, 0.9] 
f[:, 1, 1] = [ 0.05, 0.9, 0.05] 
fact = Factor(f) 
 
vn_LVEDVolume = VariableNode('vn:LVEDVolume', 3) 
fn_LVEDVolume_Hypovolemia_LVFailure = FactorNode('fn:LVEDVolume_Hypovolemia_LVFailure', fact)
 
vn_LVEDVolume.addNode(fn_LVEDVolume_Hypovolemia_LVFailure) 
fn_LVEDVolume_Hypovolemia_LVFailure.addNode(vn_LVEDVolume) 
vn_Hypovolemia.addNode(fn_LVEDVolume_Hypovolemia_LVFailure) 
fn_LVEDVolume_Hypovolemia_LVFailure.addNode(vn_Hypovolemia) 
vn_LVFailure.addNode(fn_LVEDVolume_Hypovolemia_LVFailure) 
fn_LVEDVolume_Hypovolemia_LVFailure.addNode(vn_LVFailure) 

# now we do the next node 
 
f = np.zeros((3, 2, 2))
f[:, 0, 0] = [ 0.98, 0.01, 0.01] 
f[:, 0, 1] = [ 0.5, 0.49, 0.01] 
f[:, 1, 0] = [ 0.95, 0.04, 0.01] 
f[:, 1, 1] = [ 0.05, 0.9, 0.05] 
fact = Factor(f) 
 
vn_StrokeVolume = VariableNode('vn:StrokeVolume', 3)
fn_StrokeVolume_LVFailure_Hypovolemia = FactorNode('fn:StrokeVolume_LVFailure_Hypovolemia', fact)
 
vn_StrokeVolume.addNode(fn_StrokeVolume_LVFailure_Hypovolemia) 
fn_StrokeVolume_LVFailure_Hypovolemia.addNode(vn_StrokeVolume) 
vn_LVFailure.addNode(fn_StrokeVolume_LVFailure_Hypovolemia) 
fn_StrokeVolume_LVFailure_Hypovolemia.addNode(vn_LVFailure) 
vn_Hypovolemia.addNode(fn_StrokeVolume_LVFailure_Hypovolemia) 
fn_StrokeVolume_LVFailure_Hypovolemia.addNode(vn_Hypovolemia) 

# now we do the next node 
 
f = np.zeros((3, 3))
f[:, 0] = [ 0.95, 0.04, 0.01] 
f[:, 1] = [ 0.04, 0.95, 0.01] 
f[:, 2] = [ 0.01, 0.29, 0.7] 
fact = Factor(f) 
 
vn_CVP = VariableNode('vn:CVP', 3) 
fn_CVP_LVEDVolume = FactorNode('fn:CVP_LVEDVolume', fact)
 
vn_CVP.addNode(fn_CVP_LVEDVolume) 
fn_CVP_LVEDVolume.addNode(vn_CVP)
vn_LVEDVolume.addNode(fn_CVP_LVEDVolume) 
fn_CVP_LVEDVolume.addNode(vn_LVEDVolume) 

# now we do the next node 
 
f = np.zeros((3, 3))
f[:, 0] = [ 0.95, 0.04, 0.01] 
f[:, 1] = [ 0.04, 0.95, 0.01] 
f[:, 2] = [ 0.01, 0.04, 0.95] 
fact = Factor(f) 
 
vn_PCWP = VariableNode('vn:PCWP', 3)
fn_PCWP_LVEDVolume = FactorNode('fn:PCWP_LVEDVolume', fact)
 
vn_PCWP.addNode(fn_PCWP_LVEDVolume) 
fn_PCWP_LVEDVolume.addNode(vn_PCWP)
vn_LVEDVolume.addNode(fn_PCWP_LVEDVolume) 
fn_PCWP_LVEDVolume.addNode(vn_LVEDVolume) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.2, 0.8] 
fact = Factor(f) 
 
vn_InsuffAnesth = VariableNode('vn:InsuffAnesth', 2)
fn_InsuffAnesth = FactorNode('fn:InsuffAnesth', fact)
 
vn_InsuffAnesth.addNode(fn_InsuffAnesth) 
fn_InsuffAnesth.addNode(vn_InsuffAnesth) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.01, 0.99] 
fact = Factor(f) 
 
vn_PulmEmbolus = VariableNode('vn:PulmEmbolus', 2)
fn_PulmEmbolus = FactorNode('fn:PulmEmbolus', fact)
 
vn_PulmEmbolus.addNode(fn_PulmEmbolus) 
fn_PulmEmbolus.addNode(vn_PulmEmbolus) 

# now we do the next node 
 
f = np.zeros((3, 1))
f[:, 0] = [ 0.92, 0.03, 0.05] 
fact = Factor(f) 
 
vn_Intubation = VariableNode('vn:Intubation', 3) 
fn_Intubation = FactorNode('fn:Intubation', fact)
 
vn_Intubation.addNode(fn_Intubation) 
fn_Intubation.addNode(vn_Intubation) 

# now we do the next node 
 
f = np.zeros((2, 2, 3))
f[:, 0, 0] = [ 0.1, 0.9] 
f[:, 0, 1] = [ 0.1, 0.9] 
f[:, 0, 2] = [ 0.01, 0.99] 
f[:, 1, 0] = [ 0.95, 0.05] 
f[:, 1, 1] = [ 0.95, 0.05] 
f[:, 1, 2] = [ 0.05, 0.95] 
fact = Factor(f) 
 
vn_Shunt = VariableNode('vn:Shunt', 2)
fn_Shunt_PulmEmbolus_Intubation = FactorNode('fn:Shunt_PulmEmbolus_Intubation', fact)
 
vn_Shunt.addNode(fn_Shunt_PulmEmbolus_Intubation) 
fn_Shunt_PulmEmbolus_Intubation.addNode(vn_Shunt)
vn_PulmEmbolus.addNode(fn_Shunt_PulmEmbolus_Intubation) 
fn_Shunt_PulmEmbolus_Intubation.addNode(vn_PulmEmbolus) 
vn_Intubation.addNode(fn_Shunt_PulmEmbolus_Intubation) 
fn_Shunt_PulmEmbolus_Intubation.addNode(vn_Intubation) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.04, 0.96] 
fact = Factor(f) 
 
vn_KinkedTube = VariableNode('vn:KinkedTube', 2)
fn_KinkedTube = FactorNode('fn:KinkedTube', fact)
 
vn_KinkedTube.addNode(fn_KinkedTube) 
fn_KinkedTube.addNode(vn_KinkedTube) 

# now we do the next node 
 
f = np.zeros((3, 1))
f[:, 0] = [ 0.01, 0.98, 0.01] 
fact = Factor(f) 
 
vn_MinVolSet = VariableNode('vn:MinVolSet', 3)
fn_MinVolSet = FactorNode('fn:MinVolSet', fact)
 
vn_MinVolSet.addNode(fn_MinVolSet)
fn_MinVolSet.addNode(vn_MinVolSet)

# now we do the next node 
 
f = np.zeros((4, 3))
f[:, 0] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 2] = [ 0.01, 0.01, 0.01, 0.97] 
fact = Factor(f) 
 
vn_VentMach = VariableNode('vn:VentMach', 4) 
fn_VentMach_MinVolSet = FactorNode('fn:VentMach_MinVolSet', fact)
 
vn_VentMach.addNode(fn_VentMach_MinVolSet)
fn_VentMach_MinVolSet.addNode(vn_VentMach) 
vn_MinVolSet.addNode(fn_VentMach_MinVolSet)
fn_VentMach_MinVolSet.addNode(vn_MinVolSet)

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.05, 0.95] 
fact = Factor(f) 
 
vn_Disconnect = VariableNode('vn:Disconnect', 2)
fn_Disconnect = FactorNode('fn:Disconnect', fact)
 
vn_Disconnect.addNode(fn_Disconnect)
fn_Disconnect.addNode(vn_Disconnect)

# now we do the next node 
 
f = np.zeros((4, 4, 2))
f[:, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 1] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 2, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 2, 1] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 3, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 3, 1] = [ 0.01, 0.01, 0.01, 0.97] 
fact = Factor(f) 
 
vn_VentTube = VariableNode('vn:VentTube', 4)
fn_VentTube_VentMach_Disconnect = FactorNode('fn:VentTube_VentMach_Disconnect', fact)
 
vn_VentTube.addNode(fn_VentTube_VentMach_Disconnect)
fn_VentTube_VentMach_Disconnect.addNode(vn_VentTube) 
vn_VentMach.addNode(fn_VentTube_VentMach_Disconnect)
fn_VentTube_VentMach_Disconnect.addNode(vn_VentMach) 
vn_Disconnect.addNode(fn_VentTube_VentMach_Disconnect)
fn_VentTube_VentMach_Disconnect.addNode(vn_Disconnect)

# now we do the next node 
 
f = np.zeros((4, 2, 4, 3))
f[:, 0, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 0, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 0, 2] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1, 0] = [ 0.95, 0.03, 0.01, 0.01] 
f[:, 0, 1, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1, 2] = [ 0.95, 0.03, 0.01, 0.01] 
f[:, 0, 2, 0] = [ 0.4, 0.58, 0.01, 0.01] 
f[:, 0, 2, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 2, 2] = [ 0.5, 0.48, 0.01, 0.01] 
f[:, 0, 3, 0] = [ 0.3, 0.68, 0.01, 0.01] 
f[:, 0, 3, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 3, 2] = [ 0.3, 0.68, 0.01, 0.01] 
f[:, 1, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 0, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 0, 2] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 1, 0] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1, 1, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 1, 2] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1, 2, 0] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 1, 2, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 2, 2] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 1, 3, 0] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 1, 3, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 3, 2] = [ 0.01, 0.01, 0.01, 0.97] 
fact = Factor(f) 
 
vn_VentLung = VariableNode('vn:VentLung', 4)
fn_VentLung_KinkedTube_VentTube_Intubation = FactorNode('fn:VentLung_KinkedTube_VentTube_Intubation', fact)
 
vn_VentLung.addNode(fn_VentLung_KinkedTube_VentTube_Intubation) 
fn_VentLung_KinkedTube_VentTube_Intubation.addNode(vn_VentLung) 
vn_KinkedTube.addNode(fn_VentLung_KinkedTube_VentTube_Intubation) 
fn_VentLung_KinkedTube_VentTube_Intubation.addNode(vn_KinkedTube) 
vn_VentTube.addNode(fn_VentLung_KinkedTube_VentTube_Intubation) 
fn_VentLung_KinkedTube_VentTube_Intubation.addNode(vn_VentTube) 
vn_Intubation.addNode(fn_VentLung_KinkedTube_VentTube_Intubation) 
fn_VentLung_KinkedTube_VentTube_Intubation.addNode(vn_Intubation) 

# now we do the next node 
 
f = np.zeros((4, 3, 4))
f[:, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 0, 2] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 0, 3] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 1, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 1] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1, 2] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 1, 3] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 2, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 2, 1] = [ 0.03, 0.95, 0.01, 0.01] 
f[:, 2, 2] = [ 0.01, 0.94, 0.04, 0.01] 
f[:, 2, 3] = [ 0.01, 0.88, 0.1, 0.01] 
fact = Factor(f) 
 
vn_VentAlv = VariableNode('vn:VentAlv', 4)
fn_VentAlv_Intubation_VentLung = FactorNode('fn:VentAlv_Intubation_VentLung', fact)
 
vn_VentAlv.addNode(fn_VentAlv_Intubation_VentLung) 
fn_VentAlv_Intubation_VentLung.addNode(vn_VentAlv) 
vn_Intubation.addNode(fn_VentAlv_Intubation_VentLung) 
fn_VentAlv_Intubation_VentLung.addNode(vn_Intubation) 
vn_VentLung.addNode(fn_VentAlv_Intubation_VentLung) 
fn_VentAlv_Intubation_VentLung.addNode(vn_VentLung) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.01, 0.99] 
fact = Factor(f) 
 
vn_FiO2 = VariableNode('vn:FiO2', 2)
fn_FiO2 = FactorNode('fn:FiO2', fact)
 
vn_FiO2.addNode(fn_FiO2)
fn_FiO2.addNode(vn_FiO2) 

# now we do the next node 
 
f = np.zeros((3, 4, 2))
f[:, 0, 0] = [ 0.98, 0.01, 0.01] 
f[:, 0, 1] = [ 0.98, 0.01, 0.01] 
f[:, 1, 0] = [ 0.98, 0.01, 0.01] 
f[:, 1, 1] = [ 0.98, 0.01, 0.01] 
f[:, 2, 0] = [ 0.95, 0.04, 0.01] 
f[:, 2, 1] = [ 0.01, 0.95, 0.04] 
f[:, 3, 0] = [ 0.95, 0.04, 0.01] 
f[:, 3, 1] = [ 0.01, 0.01, 0.98] 
fact = Factor(f) 
 
vn_PVSat = VariableNode('vn:PVSat', 3) 
fn_PVSat_VentAlv_FiO2 = FactorNode('fn:PVSat_VentAlv_FiO2', fact)
 
vn_PVSat.addNode(fn_PVSat_VentAlv_FiO2) 
fn_PVSat_VentAlv_FiO2.addNode(vn_PVSat)
vn_VentAlv.addNode(fn_PVSat_VentAlv_FiO2) 
fn_PVSat_VentAlv_FiO2.addNode(vn_VentAlv) 
vn_FiO2.addNode(fn_PVSat_VentAlv_FiO2) 
fn_PVSat_VentAlv_FiO2.addNode(vn_FiO2) 

# now we do the next node 
 
f = np.zeros((3, 2, 3))
f[:, 0, 0] = [ 0.98, 0.01, 0.01] 
f[:, 0, 1] = [ 0.01, 0.98, 0.01] 
f[:, 0, 2] = [ 0.01, 0.01, 0.98] 
f[:, 1, 0] = [ 0.98, 0.01, 0.01] 
f[:, 1, 1] = [ 0.98, 0.01, 0.01] 
f[:, 1, 2] = [ 0.69, 0.3, 0.01] 
fact = Factor(f) 
 
vn_SaO2 = VariableNode('vn:SaO2', 3)
fn_SaO2_Shunt_PVSat = FactorNode('fn:SaO2_Shunt_PVSat', fact)
 
vn_SaO2.addNode(fn_SaO2_Shunt_PVSat)
fn_SaO2_Shunt_PVSat.addNode(vn_SaO2)
vn_Shunt.addNode(fn_SaO2_Shunt_PVSat)
fn_SaO2_Shunt_PVSat.addNode(vn_Shunt)
vn_PVSat.addNode(fn_SaO2_Shunt_PVSat)
fn_SaO2_Shunt_PVSat.addNode(vn_PVSat)

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.01, 0.99] 
fact = Factor(f) 
 
vn_Anaphylaxis = VariableNode('vn:Anaphylaxis', 2) 
fn_Anaphylaxis = FactorNode('fn:Anaphylaxis', fact)
 
vn_Anaphylaxis.addNode(fn_Anaphylaxis) 
fn_Anaphylaxis.addNode(vn_Anaphylaxis) 

# now we do the next node 
 
f = np.zeros((3, 2))
f[:, 0] = [ 0.98, 0.01, 0.01] 
f[:, 1] = [ 0.3, 0.4, 0.3] 
fact = Factor(f) 
 
vn_TPR = VariableNode('vn:TPR', 3) 
fn_TPR_Anaphylaxis = FactorNode('fn:TPR_Anaphylaxis', fact)
 
vn_TPR.addNode(fn_TPR_Anaphylaxis) 
fn_TPR_Anaphylaxis.addNode(vn_TPR) 
vn_Anaphylaxis.addNode(fn_TPR_Anaphylaxis) 
fn_TPR_Anaphylaxis.addNode(vn_Anaphylaxis) 

# now we do the next node 
 
f = np.zeros((3, 4)) 
f[:, 0] = [ 0.01, 0.01, 0.98] 
f[:, 1] = [ 0.01, 0.01, 0.98] 
f[:, 2] = [ 0.04, 0.92, 0.04] 
f[:, 3] = [ 0.9, 0.09, 0.01] 
fact = Factor(f) 
 
vn_ArtCO2 = VariableNode('vn:ArtCO2', 3) 
fn_ArtCO2_VentAlv = FactorNode('fn:ArtCO2_VentAlv', fact)
 
vn_ArtCO2.addNode(fn_ArtCO2_VentAlv) 
fn_ArtCO2_VentAlv.addNode(vn_ArtCO2)
vn_VentAlv.addNode(fn_ArtCO2_VentAlv) 
fn_ArtCO2_VentAlv.addNode(vn_VentAlv) 

# now we do the next node 
 
f = np.zeros((2, 2, 3, 3, 3))
f[:, 0, 0, 0, 0] = [ 0.01, 0.99] 
f[:, 0, 0, 0, 1] = [ 0.01, 0.99] 
f[:, 0, 0, 0, 2] = [ 0.01, 0.99] 
f[:, 0, 0, 1, 0] = [ 0.01, 0.99] 
f[:, 0, 0, 1, 1] = [ 0.01, 0.99] 
f[:, 0, 0, 1, 2] = [ 0.01, 0.99] 
f[:, 0, 0, 2, 0] = [ 0.01, 0.99] 
f[:, 0, 0, 2, 1] = [ 0.01, 0.99] 
f[:, 0, 0, 2, 2] = [ 0.01, 0.99] 
f[:, 0, 1, 0, 0] = [ 0.01, 0.99] 
f[:, 0, 1, 0, 1] = [ 0.01, 0.99] 
f[:, 0, 1, 0, 2] = [ 0.01, 0.99] 
f[:, 0, 1, 1, 0] = [ 0.01, 0.99] 
f[:, 0, 1, 1, 1] = [ 0.01, 0.99] 
f[:, 0, 1, 1, 2] = [ 0.01, 0.99] 
f[:, 0, 1, 2, 0] = [ 0.05, 0.95] 
f[:, 0, 1, 2, 1] = [ 0.05, 0.95] 
f[:, 0, 1, 2, 2] = [ 0.01, 0.99] 
f[:, 0, 2, 0, 0] = [ 0.01, 0.99] 
f[:, 0, 2, 0, 1] = [ 0.01, 0.99] 
f[:, 0, 2, 0, 2] = [ 0.01, 0.99] 
f[:, 0, 2, 1, 0] = [ 0.05, 0.95] 
f[:, 0, 2, 1, 1] = [ 0.05, 0.95] 
f[:, 0, 2, 1, 2] = [ 0.01, 0.99] 
f[:, 0, 2, 2, 0] = [ 0.05, 0.95] 
f[:, 0, 2, 2, 1] = [ 0.05, 0.95] 
f[:, 0, 2, 2, 2] = [ 0.01, 0.99] 
f[:, 1, 0, 0, 0] = [ 0.05, 0.95] 
f[:, 1, 0, 0, 1] = [ 0.05, 0.95] 
f[:, 1, 0, 0, 2] = [ 0.01, 0.99] 
f[:, 1, 0, 1, 0] = [ 0.05, 0.95] 
f[:, 1, 0, 1, 1] = [ 0.05, 0.95] 
f[:, 1, 0, 1, 2] = [ 0.01, 0.99] 
f[:, 1, 0, 2, 0] = [ 0.05, 0.95] 
f[:, 1, 0, 2, 1] = [ 0.05, 0.95] 
f[:, 1, 0, 2, 2] = [ 0.01, 0.99] 
f[:, 1, 1, 0, 0] = [ 0.1, 0.9] 
f[:, 1, 1, 0, 1] = [ 0.1, 0.9] 
f[:, 1, 1, 0, 2] = [ 0.1, 0.9] 
f[:, 1, 1, 1, 0] = [ 0.95, 0.05] 
f[:, 1, 1, 1, 1] = [ 0.95, 0.05] 
f[:, 1, 1, 1, 2] = [ 0.3, 0.7] 
f[:, 1, 1, 2, 0] = [ 0.95, 0.05] 
f[:, 1, 1, 2, 1] = [ 0.95, 0.05] 
f[:, 1, 1, 2, 2] = [ 0.3, 0.7] 
f[:, 1, 2, 0, 0] = [ 0.95, 0.05] 
f[:, 1, 2, 0, 1] = [ 0.95, 0.05] 
f[:, 1, 2, 0, 2] = [ 0.3, 0.7] 
f[:, 1, 2, 1, 0] = [ 0.99, 0.00999999] 
f[:, 1, 2, 1, 1] = [ 0.99, 0.00999999] 
f[:, 1, 2, 1, 2] = [ 0.99, 0.00999999] 
f[:, 1, 2, 2, 0] = [ 0.95, 0.05] 
f[:, 1, 2, 2, 1] = [ 0.99, 0.00999999] 
f[:, 1, 2, 2, 2] = [ 0.3, 0.7] 
fact = Factor(f) 
 
vn_Catechol = VariableNode('vn:Catechol', 2) 
fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2 = FactorNode('fn:Catechol_InsuffAnesth_SaO2_TPR_ArtCO2', fact)
 
vn_Catechol.addNode(fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2)
fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2.addNode(vn_Catechol) 
vn_InsuffAnesth.addNode(fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2)
fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2.addNode(vn_InsuffAnesth) 
vn_SaO2.addNode(fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2)
fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2.addNode(vn_SaO2) 
vn_TPR.addNode(fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2)
fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2.addNode(vn_TPR) 
vn_ArtCO2.addNode(fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2)
fn_Catechol_InsuffAnesth_SaO2_TPR_ArtCO2.addNode(vn_ArtCO2)

# now we do the next node 
 
f = np.zeros((3, 2))
f[:, 0] = [ 0.1, 0.89, 0.01] 
f[:, 1] = [ 0.01, 0.09, 0.9] 
fact = Factor(f) 
 
vn_HR = VariableNode('vn:HR', 3) 
fn_HR_Catechol = FactorNode('fn:HR_Catechol', fact)
 
vn_HR.addNode(fn_HR_Catechol) 
fn_HR_Catechol.addNode(vn_HR) 
vn_Catechol.addNode(fn_HR_Catechol) 
fn_HR_Catechol.addNode(vn_Catechol) 

# now we do the next node 
 
f = np.zeros((3, 3, 3))
f[:, 0, 0] = [ 0.98, 0.01, 0.01] 
f[:, 0, 1] = [ 0.95, 0.04, 0.01] 
f[:, 0, 2] = [ 0.3, 0.69, 0.01] 
f[:, 1, 0] = [ 0.95, 0.04, 0.01] 
f[:, 1, 1] = [ 0.04, 0.95, 0.01] 
f[:, 1, 2] = [ 0.01, 0.3, 0.69] 
f[:, 2, 0] = [ 0.8, 0.19, 0.01] 
f[:, 2, 1] = [ 0.01, 0.04, 0.95] 
f[:, 2, 2] = [ 0.01, 0.01, 0.98] 
fact = Factor(f) 
 
vn_CO = VariableNode('vn:CO', 3) 
fn_CO_HR_StrokeVolume = FactorNode('fn:CO_HR_StrokeVolume', fact)
 
vn_CO.addNode(fn_CO_HR_StrokeVolume) 
fn_CO_HR_StrokeVolume.addNode(vn_CO) 
vn_HR.addNode(fn_CO_HR_StrokeVolume) 
fn_CO_HR_StrokeVolume.addNode(vn_HR) 
vn_StrokeVolume.addNode(fn_CO_HR_StrokeVolume) 
fn_CO_HR_StrokeVolume.addNode(vn_StrokeVolume) 

# now we do the next node 
 
f = np.zeros((2, 2))
f[:, 0] = [ 0.9, 0.1] 
f[:, 1] = [ 0.01, 0.99] 
fact = Factor(f) 
 
vn_History = VariableNode('vn:History', 2) 
fn_History_LVFailure = FactorNode('fn:History_LVFailure', fact)
 
vn_History.addNode(fn_History_LVFailure) 
fn_History_LVFailure.addNode(vn_History)
vn_LVFailure.addNode(fn_History_LVFailure) 
fn_History_LVFailure.addNode(vn_LVFailure) 

# now we do the next node 
 
f = np.zeros((3, 3, 3))
f[:, 0, 0] = [ 0.98, 0.01, 0.01] 
f[:, 0, 1] = [ 0.98, 0.01, 0.01] 
f[:, 0, 2] = [ 0.3, 0.6, 0.1] 
f[:, 1, 0] = [ 0.98, 0.01, 0.01] 
f[:, 1, 1] = [ 0.1, 0.85, 0.05] 
f[:, 1, 2] = [ 0.05, 0.4, 0.55] 
f[:, 2, 0] = [ 0.9, 0.09, 0.01] 
f[:, 2, 1] = [ 0.05, 0.2, 0.75] 
f[:, 2, 2] = [ 0.01, 0.09, 0.9] 
fact = Factor(f) 
 
vn_BP = VariableNode('vn:BP', 3) 
fn_BP_CO_TPR = FactorNode('fn:BP_CO_TPR', fact)
 
vn_BP.addNode(fn_BP_CO_TPR) 
fn_BP_CO_TPR.addNode(vn_BP) 
vn_CO.addNode(fn_BP_CO_TPR) 
fn_BP_CO_TPR.addNode(vn_CO) 
vn_TPR.addNode(fn_BP_CO_TPR) 
fn_BP_CO_TPR.addNode(vn_TPR) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.1, 0.9] 
fact = Factor(f) 
 
vn_ErrCauter = VariableNode('vn:ErrCauter', 2) 
fn_ErrCauter = FactorNode('fn:ErrCauter', fact)
 
vn_ErrCauter.addNode(fn_ErrCauter) 
fn_ErrCauter.addNode(vn_ErrCauter) 

# now we do the next node 
 
f = np.zeros((3, 3, 2))
f[:, 0, 0] = [ 0.3333333, 0.3333333, 0.3333333] 
f[:, 0, 1] = [ 0.98, 0.01, 0.01] 
f[:, 1, 0] = [ 0.3333333, 0.3333333, 0.3333333] 
f[:, 1, 1] = [ 0.01, 0.98, 0.01] 
f[:, 2, 0] = [ 0.3333333, 0.3333333, 0.3333333] 
f[:, 2, 1] = [ 0.01, 0.01, 0.98] 
fact = Factor(f) 
 
vn_HREKG = VariableNode('vn:HREKG', 3)
fn_HREKG_HR_ErrCauter = FactorNode('fn:HREKG_HR_ErrCauter', fact)
 
vn_HREKG.addNode(fn_HREKG_HR_ErrCauter) 
fn_HREKG_HR_ErrCauter.addNode(vn_HREKG) 
vn_HR.addNode(fn_HREKG_HR_ErrCauter) 
fn_HREKG_HR_ErrCauter.addNode(vn_HR) 
vn_ErrCauter.addNode(fn_HREKG_HR_ErrCauter) 
fn_HREKG_HR_ErrCauter.addNode(vn_ErrCauter) 

# now we do the next node 
 
f = np.zeros((3, 3, 2)) 
f[:, 0, 0] = [ 0.3333333, 0.3333333, 0.3333333] 
f[:, 0, 1] = [ 0.98, 0.01, 0.01] 
f[:, 1, 0] = [ 0.3333333, 0.3333333, 0.3333333] 
f[:, 1, 1] = [ 0.01, 0.98, 0.01] 
f[:, 2, 0] = [ 0.3333333, 0.3333333, 0.3333333] 
f[:, 2, 1] = [ 0.01, 0.01, 0.98] 
fact = Factor(f) 
 
vn_HRSat = VariableNode('vn:HRSat', 3)
fn_HRSat_HR_ErrCauter = FactorNode('fn:HRSat_HR_ErrCauter', fact)
 
vn_HRSat.addNode(fn_HRSat_HR_ErrCauter)
fn_HRSat_HR_ErrCauter.addNode(vn_HRSat)
vn_HR.addNode(fn_HRSat_HR_ErrCauter)
fn_HRSat_HR_ErrCauter.addNode(vn_HR)
vn_ErrCauter.addNode(fn_HRSat_HR_ErrCauter) 
fn_HRSat_HR_ErrCauter.addNode(vn_ErrCauter) 

# now we do the next node 
 
f = np.zeros((2, 1))
f[:, 0] = [ 0.05, 0.95] 
fact = Factor(f) 
 
vn_ErrLowOutput = VariableNode('vn:ErrLowOutput', 2)
fn_ErrLowOutput = FactorNode('fn:ErrLowOutput', fact)
 
vn_ErrLowOutput.addNode(fn_ErrLowOutput)
fn_ErrLowOutput.addNode(vn_ErrLowOutput)

# now we do the next node 
 
f = np.zeros((3, 2, 3)) 
f[:, 0, 0] = [ 0.98, 0.01, 0.01] 
f[:, 0, 1] = [ 0.4, 0.59, 0.01] 
f[:, 0, 2] = [ 0.3, 0.4, 0.3] 
f[:, 1, 0] = [ 0.98, 0.01, 0.01] 
f[:, 1, 1] = [ 0.01, 0.98, 0.01] 
f[:, 1, 2] = [ 0.01, 0.01, 0.98] 
fact = Factor(f) 
 
vn_HRBP = VariableNode('vn:HRBP', 3)
fn_HRBP_ErrLowOutput_HR = FactorNode('fn:HRBP_ErrLowOutput_HR', fact)
 
vn_HRBP.addNode(fn_HRBP_ErrLowOutput_HR)
fn_HRBP_ErrLowOutput_HR.addNode(vn_HRBP)
vn_ErrLowOutput.addNode(fn_HRBP_ErrLowOutput_HR)
fn_HRBP_ErrLowOutput_HR.addNode(vn_ErrLowOutput) 
vn_HR.addNode(fn_HRBP_ErrLowOutput_HR)
fn_HRBP_ErrLowOutput_HR.addNode(vn_HR) 

# now we do the next node 
 
f = np.zeros((4, 3, 4))
f[:, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 0, 2] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 0, 3] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 1] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 1, 2] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 1, 3] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 2, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 2, 1] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 2, 2] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 2, 3] = [ 0.01, 0.01, 0.01, 0.97] 
fact = Factor(f) 
 
vn_ExpCO2 = VariableNode('vn:ExpCO2', 4)
fn_ExpCO2_ArtCO2_VentLung = FactorNode('fn:ExpCO2_ArtCO2_VentLung', fact)
 
vn_ExpCO2.addNode(fn_ExpCO2_ArtCO2_VentLung) 
fn_ExpCO2_ArtCO2_VentLung.addNode(vn_ExpCO2) 
vn_ArtCO2.addNode(fn_ExpCO2_ArtCO2_VentLung) 
fn_ExpCO2_ArtCO2_VentLung.addNode(vn_ArtCO2) 
vn_VentLung.addNode(fn_ExpCO2_ArtCO2_VentLung) 
fn_ExpCO2_ArtCO2_VentLung.addNode(vn_VentLung) 

# now we do the next node 
 
f = np.zeros((3, 2))
f[:, 0] = [ 0.01, 0.19, 0.8] 
f[:, 1] = [ 0.05, 0.9, 0.05] 
fact = Factor(f) 
 
vn_PAP = VariableNode('vn:PAP', 3) 
fn_PAP_PulmEmbolus = FactorNode('fn:PAP_PulmEmbolus', fact)
 
vn_PAP.addNode(fn_PAP_PulmEmbolus) 
fn_PAP_PulmEmbolus.addNode(vn_PAP)
vn_PulmEmbolus.addNode(fn_PAP_PulmEmbolus) 
fn_PAP_PulmEmbolus.addNode(vn_PulmEmbolus) 

# now we do the next node 
 
f = np.zeros((4, 2, 3, 4))
f[:, 0, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 0, 1] = [ 0.01, 0.49, 0.3, 0.2] 
f[:, 0, 0, 2] = [ 0.01, 0.01, 0.08, 0.9] 
f[:, 0, 0, 3] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 0, 1, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1, 1] = [ 0.1, 0.84, 0.05, 0.01] 
f[:, 0, 1, 2] = [ 0.05, 0.25, 0.25, 0.45] 
f[:, 0, 1, 3] = [ 0.01, 0.15, 0.25, 0.59] 
f[:, 0, 2, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 2, 1] = [ 0.01, 0.29, 0.3, 0.4] 
f[:, 0, 2, 2] = [ 0.01, 0.01, 0.08, 0.9] 
f[:, 0, 2, 3] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 1, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 0, 1] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1, 0, 2] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 1, 0, 3] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 1, 1, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 1, 1] = [ 0.4, 0.58, 0.01, 0.01] 
f[:, 1, 1, 2] = [ 0.2, 0.75, 0.04, 0.01] 
f[:, 1, 1, 3] = [ 0.2, 0.7, 0.09, 0.01] 
f[:, 1, 2, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 2, 1] = [ 0.01, 0.9, 0.08, 0.01] 
f[:, 1, 2, 2] = [ 0.01, 0.01, 0.38, 0.6] 
f[:, 1, 2, 3] = [ 0.01, 0.01, 0.01, 0.97] 
fact = Factor(f) 
 
vn_Press = VariableNode('vn:Press', 4)
fn_Press_KinkedTube_Intubation_VentTube = FactorNode('fn:Press_KinkedTube_Intubation_VentTube', fact)
 
vn_Press.addNode(fn_Press_KinkedTube_Intubation_VentTube) 
fn_Press_KinkedTube_Intubation_VentTube.addNode(vn_Press) 
vn_KinkedTube.addNode(fn_Press_KinkedTube_Intubation_VentTube) 
fn_Press_KinkedTube_Intubation_VentTube.addNode(vn_KinkedTube) 
vn_Intubation.addNode(fn_Press_KinkedTube_Intubation_VentTube) 
fn_Press_KinkedTube_Intubation_VentTube.addNode(vn_Intubation) 
vn_VentTube.addNode(fn_Press_KinkedTube_Intubation_VentTube) 
fn_Press_KinkedTube_Intubation_VentTube.addNode(vn_VentTube) 

# now we do the next node 
 
f = np.zeros((4, 4, 3))
f[:, 0, 0] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 1] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 0, 2] = [ 0.97, 0.01, 0.01, 0.01] 
f[:, 1, 0] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 1, 1] = [ 0.6, 0.38, 0.01, 0.01] 
f[:, 1, 2] = [ 0.01, 0.97, 0.01, 0.01] 
f[:, 2, 0] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 2, 1] = [ 0.5, 0.48, 0.01, 0.01] 
f[:, 2, 2] = [ 0.01, 0.01, 0.97, 0.01] 
f[:, 3, 0] = [ 0.01, 0.01, 0.01, 0.97] 
f[:, 3, 1] = [ 0.5, 0.48, 0.01, 0.01] 
f[:, 3, 2] = [ 0.01, 0.01, 0.01, 0.97] 
fact = Factor(f) 
 
vn_MinVol = VariableNode('vn:MinVol', 4)
fn_MinVol_VentLung_Intubation = FactorNode('fn:MinVol_VentLung_Intubation', fact)
 
vn_MinVol.addNode(fn_MinVol_VentLung_Intubation) 
fn_MinVol_VentLung_Intubation.addNode(vn_MinVol) 
vn_VentLung.addNode(fn_MinVol_VentLung_Intubation) 
fn_MinVol_VentLung_Intubation.addNode(vn_VentLung) 
vn_Intubation.addNode(fn_MinVol_VentLung_Intubation) 
fn_MinVol_VentLung_Intubation.addNode(vn_Intubation) 
 
# set the value of any nodes which are observed

vn_SaO2.setValue(np.array([1, 0, 0])) 
vn_BP.setValue(np.array([1, 0, 0]))
vn_ArtCO2.setValue(np.array([1, 0, 0])) 
vn_Press.setValue(np.array([0, 1, 0, 0]))
vn_ExpCO2.setValue(np.array([1, 0, 0, 0]))


# do loopy belief propagation as an inference procedure. pass messages in
# every node 20 times.

for i in range(20):
    print('i = ' + str(i))
    vn_MinVol.loopy_bp()
    vn_MinVol.setNotUpdated()

# display the marginal distribution in the variable node for Kinked Tube, Vent Lung & Anaphy Laxis
print(vn_KinkedTube.getMarginalDistribution())
print(vn_VentLung.getMarginalDistribution())
print(vn_Anaphylaxis.getMarginalDistribution())
