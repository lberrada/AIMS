""" Comverted from matlab code
Source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_1_sum_product/
"""

import numpy as np
import itertools

class Factor:
    """% factor : this class is a container class for the factor matrix
    %
    % properties
    %
    % factor_matrix : a tensor which given the factor value for any
    %                 configuration of the neighboring variable nodes
    %"""
    
    def __init__(self, factor_matrix):
        """% factor : constructor method for object
        %
        % @param factor_matrix : factor matrix
        %"""
            
        self.factor_matrix = factor_matrix;
        
    def product(self, product_of_messages):
        """% product_matrix : method to get the result of an element wise
        %                  product of an input tensor and the factor matrix
        %
        % @param product_of_messages : tensor to product with factor matrix
        %"""

        product_matrix = product_of_messages * self.factor_matrix
        return product_matrix
        
    def getValue(self, indices):
        """% getValue : method to get the value of a given element of the
        %            factor matrix
        %
        % @param varargin : variable length input array of indices into
        %                   factor matrix.  varargin should have the same
        %                   number of values as the factor matrix has
        %                   dimensions.
        %"""

        return self.factor_matrix[indices]
            
    def display(self):
        """% display : overides the default display behavior
        %"""
        
        print(self.factor_matrix)
        
        
class Node:
    
    """% node : this object is the basic element of a graph.  for inference on
    %        factor graphs we will need both factor nodes and variable nodes.  
    %        both of those types of objects will be descended from this object.
    %
    % properties
    %
    % unid      : unique string identifier for the node
    % nodes     : cell array of neighboring nodes
    % messages  : cell array of messages from neighbor nodes.  all messages are
    %             column vectors
    % updated   : indicator variable for scheduling of message passing in loopy
    %             beleif propagation
    %"""
    
    def __init__(self, unid=None):
        self.unid = unid
        self.nodes = []
        self.messages = []
        self.updated = 0

    def setNotUpdated(self):
        """% setNotUpdated : recursive method to set this the updated field of
        %                 this node and recursively all nodes in the graph 
        %                 to 0
        %"""
        self.updated = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].updated:
                self.nodes[i].setNotUpdated
        
    def loopy_bp(self):
        """% loopy_bp : recursive method to prompt this node and recursively
        %            all nodes in the graph to update their received
        %            messages
        %"""
        self.updateMessages()
        self.updated = 1
        for i in range(len(self.nodes)):
            if not self.nodes[i].updated:
                self.nodes[i].loopy_bp()
    
    def updateMessages(self):   
        """ % updateMessages : method to ask all neighboring nodes for an
        %                  updated message
        %"""
        for i in range(len(self.nodes)):
            self.messages[i] = self.nodes[i].getMessage(self.unid)
        
    def passMessageIn(self, to_unid):
        """ % passMessageIn : recursive method to pass messages in from the 
        %                 edges to the given node
        %
        % @param to_unid : node to which message is being passed
        %"""
    
        ind = -1
        for i in range(len(self.nodes)):
            if self.nodes[i].unid == to_unid:
                ind = i
                break
        if ind == -1:
            raise ValueError('must be connected to each other')

        self.messages[i] = self.nodes[i].getMessage(self.unid)
        
        for i in range(len(self.nodes)):
            if i != ind:
                self.nodes[i].passMessageIn(self.unid)
        
    def passMessageOut(self, from_unid):
        """% passMessageOut : recursive method to pass messages out to the 
        %                  corners from a central node
        %
        % @param from_unid : node from which we were prompted to pass
        %                    messages out, i.e. node to which we do not 
        %                    need to pass a message
        %"""
        
        ind = -1
        for i in range(len(self.nodes)):
            if self.nodes[i].unid == from_unid:
                ind = i
                break
            
        if ind == -1:
            raise ValueError('must be connected to each other')
        
        self.messages[i] = self.nodes[i].getMessage(self.unid)

        for i in range(len(self.nodes)):
            if i != ind:
                self.nodes[i].passMessageOut(self.unid)
    
    def addNode(self, node):
        """% addNode : add a node to the neighbor node cell array
        %"""
            
        raise ValueError('this function is meant to be abstract and over written')
        
    def getMessage(self, to_unid):
        """ % getMessage : get the message to the given node
        %
        % @param to_unid : node to which the message is being passed
        %"""

        raise ValueError('this function is meant to be abstract and over written')
        
    def display(self):
        """% display : overide of default display behavior
        %"""

        print('description : ')
        print(self.unid)
        print(' ')
        print(['this node has ' + str(len(self.nodes)) + ' neighboring nodes, they are']) 
        for i in range(len(self.nodes)):
            print(self.nodes[i].unid)
            

class VariableNode(Node):
    """% variable_node
    %
    % object specifically for variable nodes.
    %
    % properties
    %
    % dimension : dimension of discrete variable represented by node
    % observed  : indicator indicating if the variable is observed
    % value     : value of the node if it has been observed
    %"""
    
    def __init__(self, unid, dimension):
        """% variable_node : method to construct the this variable node
        %
        % @param unid      : unique identifier for this node
        % @param dimension : dimension of this node 
        %"""
        
        Node.__init__(self, unid)
        self.dimension = dimension
        self.observed = False
            
        
    def addNode(self, node):
        """% addNode : method to add a neighboring node to the list of
        %           neighboring nodes
        %
        % @param node : node to add as neighbor
        %"""
        self.nodes.append(node)
        self.messages.append(np.ones(self.dimension))
        
        
    def getMessage(self, to_unid):
        """% getMessage : override of getMessage in node class.  THIS IS A
        %              METHOD WHICH MUST BE FILLED OUT BY THE STUDENT
        %
        % @param to_unid : node to which message is being passed."""
        
         
        if self.observed:
            message = self.value
        else:
            message = np.ones(self.dimension)
            for i in range(len(self.nodes)):
                if self.nodes[i].unid != to_unid:
                    message *= self.messages[i]
        
        message /= np.sum(message)
        
        return message
                    
        
#     def passMessagesIn(self):
#         """% passMessagesIn : method to pass messages from the edge to this
#         %                  node
#         %"""
# 
#         for i in range(len(self.nodes)):
#             self.messages[i] = self.nodes[i].passMessageIn(self.unid)
#             
#         
#     def passMessagesOut(self):
#         """% passMessagesOut : method to pass messages out from this node to
#         %                   the edges
#         %"""
#         for i in range(len(self.nodes)):
#             self.nodes[i].passMessageOut(self.unid)
            
        
    def getMarginalDistribution(self):
        """% getMarginalDistribution : method to get the marginal distribution
        %                           of the variable represented by this
        %                           node. THIS IS A METHOD WHICH MUST BE
        %                           FILLED OUT BY THE STUDENT.
        %"""
        
        if self.observed:
            return self.value
        else:
            marginal = np.product(self.messages, axis=0)
            marginal /= np.sum(marginal)
            return marginal
            
        
    def setValue(self, val):
        """% setValue : method to set the vaue of this variable node.
        """
        
        assert(len(val) == self.dimension)
        
        self.value = val
        self.observed = True
        
    
    def display(self):
        """% print : overrides the default print behavior
        """
        print('var dimension is = ', self.dimension)
        if self.observed:
            print('this var is observed as ' + str(self.value))
        else:
            print('this var is hidden')
            


class FactorNode(Node):
    """% factor_node : object used for factor nodes in a factor graph
    % representation of a graphical model.
    %
    % properties
    %
    % factor : factor object associated with this factor node
    %"""

    def __init__(self, unid, factor):        
        """% factor_node : constructor method to create this factor node
        %
        % @param factor : factor associated with this factor node
        %"""
            
        Node.__init__(self, unid)
        self.factor = factor
        
    def getMessage(self, to_unid):
        """% getMessage : gets the message to be sent to the given node. THIS
        %              METHOD IS ONE WITH STUDENT NEEDS TO FILL OUT.
        %
        % @param to_unid : node to which message will be sent
        %"""
        
        to_index = [node.unid for node in self.nodes].index(to_unid)
        dim_node_to = self.nodes[to_index].dimension
        n = len(self.nodes)
        l = [range(node.dimension) for node in self.nodes]
        indices = [i for i in range(n) if i != to_index]
        
        message = np.zeros(dim_node_to)
        
        for x in range(dim_node_to):
            l[to_index] = (x,)
            my_iterators = tuple(l)
            for iterator in itertools.product(*my_iterators):
                to_multiply = [self.messages[j][iterator[j]] for j in indices]
                message[x] += np.product(to_multiply) * self.factor.getValue(iterator)
        
        message /= np.sum(message)
        
        return message
        
    
    def display(self):
        """% display : overides the default display behavior
        %"""
        print('factor is = ', self.factor);
            
        
    
