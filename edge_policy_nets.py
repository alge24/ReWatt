import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from graph_env import GraphEnv
import networkx as nx
from torch.distributions import Categorical
import encoders 
import numpy as np
import pickle
import torch.optim as optim
from torch.autograd import Variable
import random
from operator import itemgetter
import argparse
import os

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout).to(device)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim)).to(device)
        else:
            self.bias = None


    def forward(self, x, adj):

        if self.dropout > 0.001:
            x = self.dropout_layer(x)


        y = torch.matmul(adj, x)


        if self.add_self:
            y += x

        y = torch.matmul(y,self.weight)

        if self.bias is not None:
            y = y + self.bias


 
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=1)

        return y



class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,
        concat=True, dropout=0.0, args=None, embedding_normalize = False):
        super(GcnEncoderGraph, self).__init__()


        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device

        self.concat = concat
        add_self = not concat
        self.num_layers = num_layers

        self.bias = True
        if args is not None:
            self.bias = args.bias
        
        

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=embedding_normalize, dropout=dropout)
        self.act = nn.ReLU().to(device)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)




    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last




            
    def forward(self, x, adj, **kwargs):
        x = self.conv_first(x, adj)
        x = self.act(x)
        x_all = [x]
        out_all = []

        out, _ = torch.max(x, dim=0,keepdim = True)


        out_all.append(out)

        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)

            x_all.append(x)
            out,_ = torch.max(x, dim=0,keepdim = True)
 

            out_all.append(out)

        x = self.conv_last(x,adj)
        #x = self.act(x)
        x_all.append(x)


        out, _ = torch.max(x, dim=0, keepdim = True)
        out_all.append(out)


        if self.concat:
            x_out = torch.cat([*x_all],dim =1)
            output = torch.cat([*out_all], dim=1)

        else:
            x_out = x
            output = out

        return x_out ,output



class PolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,pred_hidden,use_graph_state = 1,
        concat=True, dropout=0.0, args=None, embedding_normalize=False, edge_based = 0, edge_combine = 'mult', num_hidden_layers = 0):
        super(PolicyNet, self).__init__()
        self.use_graph_state = use_graph_state
        self.num_hidden_layers = num_hidden_layers

        self.edge_based = edge_based
        if edge_based:
            self.edge_combine = edge_combine
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.graph_encoder = GcnEncoderGraph(input_dim, hidden_dim, embedding_dim, num_layers,
        concat, dropout, args, embedding_normalize).to(device)
        if concat:
            self.final_dim = hidden_dim * (num_layers - 1) + embedding_dim
            


        if use_graph_state:
            if edge_based: 
                if edge_combine == 'concat':
                    
                    self.linear1_edge = nn.Linear(3*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_edge = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_edge.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_edge = nn.Linear(pred_hidden, 1).to(device)
                    
                    self.linear1_third_node = nn.Linear(4*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_third_node = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_third_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))                    
                    self.linear2_third_node = nn.Linear(pred_hidden, 1).to(device)
                else:
                    self.linear1_edge = nn.Linear(2*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_edge = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_edge.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_edge = nn.Linear(pred_hidden, 1).to(device)
                    self.linear1_choose_node = nn.Linear(3*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_choose_node = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_choose_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_choose_node = nn.Linear(pred_hidden, 1).to(device)
                    

                    self.linear1_third_node = nn.Linear(4*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_third_node = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_third_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_third_node = nn.Linear(pred_hidden, 1).to(device)
                    
                                
            
            else:

                self.linear1_first_node = nn.Linear(2*self.final_dim, pred_hidden).to(device)
                if self.num_hidden_layers>0:
                    self.linear_hidden_layers_first_node = nn.ModuleList()
                    for i in range(self.num_hidden_layers):
                        self.linear_hidden_layers_first_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                        
                self.linear2_first_node = nn.Linear(pred_hidden,1).to(device)

                self.linear1_second_node = nn.Linear(3*self.final_dim, pred_hidden).to(device)
                
                if self.num_hidden_layers>0:
                    self.linear_hidden_layers_second_node = nn.ModuleList()
                    for i in range(self.num_hidden_layers):
                        self.linear_hidden_layers_second_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                        
                self.linear2_second_node = nn.Linear(pred_hidden,1).to(device)     

                self.linear1_third_node = nn.Linear(4*self.final_dim, pred_hidden).to(device)
                if self.num_hidden_layers>0:
                    self.linear_hidden_layers_third_node = nn.ModuleList()
                    for i in range(self.num_hidden_layers):
                        self.linear_hidden_layers_third_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                self.linear2_third_node = nn.Linear(pred_hidden,1).to(device)  
            
            
            
        else:
            
            if edge_based: 
                if edge_combine == 'concat':
                    
                    self.linear1_edge = nn.Linear(2*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_edge = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_edge.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                            
                    self.linear2_edge = nn.Linear(pred_hidden, 1).to(device)
                    
                    self.linear1_third_node = nn.Linear(3*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_third_node = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_third_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_third_node = nn.Linear(pred_hidden, 1).to(device)
                else:
                    ## choosing edge 
                    ## input:  edge_emb
                    self.linear1_edge = nn.Linear(1*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_edge = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_edge.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_edge = nn.Linear(pred_hidden, 1).to(device)
                    ### choosing which node to rewire
                    #### input: edge_emb + candidate_node_emb
                    self.linear1_choose_node = nn.Linear(2*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_choose_node = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_choose_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_choose_node = nn.Linear(pred_hidden, 1).to(device)
                    
                    ### choosing the node to rewire to
                    ## input:  edge_emb + selected_node_emb + candidate_node_emb
                    self.linear1_third_node = nn.Linear(3*self.final_dim, pred_hidden).to(device)
                    if self.num_hidden_layers>0:
                        self.linear_hidden_layers_third_node = nn.ModuleList()
                        for i in range(self.num_hidden_layers):
                            self.linear_hidden_layers_third_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                    self.linear2_third_node = nn.Linear(pred_hidden, 1).to(device)
                    
            else:
                self.linear1_first_node = nn.Linear(1*self.final_dim, pred_hidden).to(device)
                if self.num_hidden_layers>0:
                    self.linear_hidden_layers_first_node = nn.ModuleList()
                    for i in range(self.num_hidden_layers):
                        self.linear_hidden_layers_first_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                self.linear2_first_node = nn.Linear(pred_hidden,1).to(device)

                self.linear1_second_node = nn.Linear(2*self.final_dim, pred_hidden).to(device)
                if self.num_hidden_layers>0:
                    self.linear_hidden_layers_second_node = nn.ModuleList()
                    for i in range(self.num_hidden_layers):
                        self.linear_hidden_layers_second_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                self.linear2_second_node = nn.Linear(pred_hidden,1).to(device)     

                self.linear1_third_node = nn.Linear(3*self.final_dim, pred_hidden).to(device)
                if self.num_hidden_layers>0:
                    self.linear_hidden_layers_third_node = nn.ModuleList()
                    for i in range(self.num_hidden_layers):
                        self.linear_hidden_layers_third_node.append(nn.Linear(pred_hidden, pred_hidden).to(device))
                self.linear2_third_node = nn.Linear(pred_hidden,1).to(device) 

        
            
            
        
        

        self.act = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
    def forward(self,x, adj, candidate_actions, selected_nodes = None, use_one=0):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        if use_one:

            if selected_nodes is None:
                self.node_emb, self.graph_emb = self.graph_encoder(x,adj)            
        else:
            self.node_emb, self.graph_emb = self.graph_encoder(x,adj)
            
            
        if not self.edge_based:
            len_candidate_actions = len(candidate_actions)
            candidate_actions_tensor = torch.LongTensor(candidate_actions).to(device)


            candidate_action_emb = torch.index_select(self.node_emb, 0,candidate_actions_tensor)
            graph_emb_repeat = self.graph_emb.repeat(len_candidate_actions,1)
            if self.use_graph_state:
                input_tensor = torch.cat((graph_emb_repeat, candidate_action_emb),1)
            else:
                input_tensor = candidate_action_emb


            if selected_nodes == None:

                out = self.act(self.linear1_first_node(input_tensor))
                
                if self.num_hidden_layers>0:
                    for i in range(self.num_hidden_layers):
                        out = self.act(self.linear_hidden_layers_first_node[i](out))

                out = self.linear2_first_node(out)
            else:
                node1_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[0]]).to(device))

                node1_emb_repeat = node1_emb.repeat(len_candidate_actions, 1)

                input_tensor = torch.cat((input_tensor, node1_emb_repeat),1)
                if len(selected_nodes) == 1:

                    out = self.act(self.linear1_second_node(input_tensor))
                    if self.num_hidden_layers>0:
                        for i in range(self.num_hidden_layers):
                            out = self.act(self.linear_hidden_layers_second_node[i](out))
                    out = self.linear2_second_node(out)
                elif len(selected_nodes) == 2:

                    node2_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[1]]).to(device))            

                    node2_emb_repeat = node2_emb.repeat(len_candidate_actions, 1)
                    input_tensor = torch.cat((input_tensor, node2_emb_repeat),1)

                    out = self.act(self.linear1_third_node(input_tensor))
                    
                    if self.num_hidden_layers>0:
                        for i in range(self.num_hidden_layers):
                            out = self.act(self.linear_hidden_layers_third_node[i](out))

                    out = self.linear2_third_node(out)
        else:
            if candidate_actions is not None:
                if self.edge_combine == 'concat' and selected_nodes is None:
                    len_candidate_actions = 2*len(candidate_actions)
                else:
                    len_candidate_actions = len(candidate_actions)  

                if self.use_graph_state:
                    graph_emb_repeat = self.graph_emb.repeat(len_candidate_actions,1)


            if selected_nodes ==None:
                edge_emb = get_edge_candidate_emb(self.node_emb, candidate_actions, self.edge_combine, device)
                if self.use_graph_state:
                    input_tensor = torch.cat((graph_emb_repeat,edge_emb), 1)
                else:
                    input_tensor = edge_emb
                out = self.act(self.linear1_edge(input_tensor))
                
                if self.num_hidden_layers>0:
                    for i in range(self.num_hidden_layers):
                        out = self.act(self.linear_hidden_layers_edge[i](out))
                        
                out = self.linear2_edge(out)
            else:

                if self.edge_combine == 'concat':
                    node1_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[0]]).to(device))
                    node2_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[1]]).to(device))
                    node1_emb_repeat = node1_emb.repeat(len_candidate_actions, 1)
                    node2_emb_repeat = node2_emb.repeat(len_candidate_actions, 1)
                    edge_emb_repeat = torch.cat((node1_emb_repeat, node2_emb_repeat), 1)
                    if self.use_graph_state:
                        input_tensor = torch.cat((graph_emb_repeat, edge_emb_repeat), 1)
                    else:
                        input_tensor = edge_emb_repeat

                    candidate_actions_tensor = torch.LongTensor(candidate_actions).to(device)
                    candidate_action_emb = torch.index_select(self.node_emb, 0,candidate_actions_tensor)

                    input_tensor = torch.cat((input_tensor, candidate_action_emb), 1)

                    out = self.act(self.linear1_third_node(input_tensor))
                    if self.num_hidden_layers>0:
                        for i in range(self.num_hidden_layers):
                            out = self.act(self.linear_hidden_layers_third_node[i](out))
                    out = self.linear2_third_node(out)
                else:
                    if candidate_actions==None:

                        node1_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[0]]).to(device))
                        node2_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[1]]).to(device))
                        nodes_emb = torch.cat((node1_emb, node2_emb),0)  
                        if self.edge_combine == 'mult':
                            edge_emb_12 = node1_emb * node2_emb
                        elif self.edge_combine == 'add':
                            edge_emb_12 = node1_emb + node2_emb
                        edge_emb_12_re = edge_emb_12.repeat(2,1)
                        if self.use_graph_state:
                            graph_emb_re = self.graph_emb.repeat(2,1)
                            input_tensor = torch.cat((graph_emb_re, edge_emb_12_re), 1)
                        else:
                            input_tensor = edge_emb_12_re
                        input_tensor = torch.cat((input_tensor, nodes_emb),1)

                        out = self.act(self.linear1_choose_node(input_tensor))
                        if self.num_hidden_layers>0:
                            for i in range(self.num_hidden_layers):
                                out = self.act(self.linear_hidden_layers_choose_node[i](out))
                        out = self.linear2_choose_node(out)
                    else:
                        node1_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[0]]).to(device))
                        node2_emb = torch.index_select(self.node_emb, 0 , torch.LongTensor([selected_nodes[1]]).to(device))
                        if self.edge_combine == 'mult':
                            edge_emb_12 = node1_emb * node2_emb
                        elif self.edge_combine == 'add':
                            edge_emb_12 = node1_emb + node2_emb
                        edge_emb_repeat = edge_emb_12.repeat(len_candidate_actions,1)
                        if self.use_graph_state:
                            input_tensor = torch.cat((graph_emb_repeat,edge_emb_repeat),1)
                        else:
                            input_tensor = edge_emb_tensor

                        selected_node_repeat = node1_emb.repeat(len_candidate_actions, 1)
                        input_tensor = torch.cat((input_tensor, selected_node_repeat),1)


                        candidate_actions_tensor = torch.LongTensor(candidate_actions).to(device)
                        candidate_action_emb = torch.index_select(self.node_emb, 0,candidate_actions_tensor)

                        input_tensor = torch.cat((input_tensor, candidate_action_emb), 1)

                        out = self.act(self.linear1_third_node(input_tensor))
                        if self.num_hidden_layers>0:
                            for i in range(self.num_hidden_layers):
                                out = self.act(self.linear_hidden_layers_third_node[i](out))
                        out = self.linear2_third_node(out)
                            
                            


        return self.softmax(out.view(1,-1)).view(-1)
    
def get_edge_candidate_emb(node_emb, candidate_actions, edge_combine, device):
    list_1 = []
    list_2 = []
    for i in range(len(candidate_actions)):
        list_1.append(candidate_actions[i][0])
        list_2.append(candidate_actions[i][1])
    tensor_1 = torch.LongTensor(list_1).to(device)
    tensor_2 = torch.LongTensor(list_2).to(device)
    
    emb_1 = torch.index_select(node_emb,0,tensor_1)
    emb_2 = torch.index_select(node_emb,0,tensor_2)
    
    if edge_combine=='concat':
        emb_l = torch.cat((emb_1, emb_2), 0)
        emb_r = torch.cat((emb_2, emb_1), 0)
        edge_emb = torch.cat((emb_l,emb_r), 1)
    elif edge_combine == 'mult':
        edge_emb = emb_1*emb_2
    elif edge_combine == 'add':
        edge_emb = emb_1 + emb_2
        
    return edge_emb
    
    


def get_candidate_nodes(graph):
    candidate_nodes = list(range(nx.number_of_nodes(graph)))
    for node in candidate_nodes:
        neighbors = set(graph.neighbors(node))
        if len(neighbors)==0:
            candidate_nodes.remove(node)
    return list(candidate_nodes)

def get_candidate_nodes_1(graph,k=1):
    candidate_nodes = list(range(nx.number_of_nodes(graph)))
    for node in candidate_nodes:
        degree = graph.degree(node)
        if degree<=k:
            candidate_nodes.remove(node)
            
    return list(candidate_nodes)

                
    
                
                
                
                
        
    
    
    

class Agent():
    def __init__(self,input_dim, hidden_dim, embedding_dim, num_layers,pred_hidden, env, nor,use_one, use_graph_state, md, args=None, embedding_normalize=False, padding=False, edge_based = 0, edge_combine='mult', num_hidden_layers = 0):
        self.policy = PolicyNet(input_dim, hidden_dim, embedding_dim, num_layers,pred_hidden, use_graph_state, args= args, embedding_normalize=embedding_normalize, edge_based=edge_based, edge_combine=edge_combine, num_hidden_layers=num_hidden_layers)
        self.env = env
        self.nor = nor
        self.use_one = use_one
        self.md = md
        self.padding = padding
        self.edge_based = edge_based
        self.edge_combine = edge_combine

            
    def select_action(self, x, state):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        adj = nx.adjacency_matrix(state).todense()

        adj = np.asarray(adj)


        if self.nor:
            deg = np.sum(adj, axis=0, dtype=np.float)
            deg[deg<0.5] = 100

            sqrt_deg = np.sqrt(deg)

            sqrt_deg = 1/sqrt_deg

            sqrt_deg = np.diag(sqrt_deg)

            adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

        if self.padding:
            new_adj = np.zeros((adj.shape[0]+1,adj.shape[0]+1))
            new_adj[:adj.shape[0],:adj.shape[0]] = adj 
            adj = new_adj



        adj_tensor = torch.from_numpy(adj).type(torch.FloatTensor).to(device)            
            
            
        if not self.edge_based:

            num_nodes = nx.number_of_nodes(state)

            candidate_actions_1 = get_candidate_nodes_1(state, self.md)



            prob_1 = self.policy(x, adj_tensor, candidate_actions_1,None,self.use_one)
            sampler_1 = Categorical(prob_1)
            index_1 = sampler_1.sample()
            node_1 = candidate_actions_1[index_1]
            selected_nodes = [node_1]
            log_prob = sampler_1.log_prob(index_1)




            #### choosing the second node

            candidate_actions_2 = self.env.constraint_action_space(selected_nodes)
            if len(candidate_actions_2) == 0:
                return None, None

            prob_2 = self.policy(x, adj_tensor, candidate_actions_2, selected_nodes, self.use_one)
            sampler_2 = Categorical(prob_2)
            index_2 = sampler_2.sample()
            node_2 = candidate_actions_2[index_2]
            selected_nodes.append(node_2)
            log_prob = log_prob + sampler_2.log_prob(index_2)



            ### choosing the third node
            candidate_actions_3 = self.env.constraint_action_space(selected_nodes)
            if len(candidate_actions_3) == 0:
                return None, None

            prob_3 = self.policy(x, adj_tensor, candidate_actions_3, selected_nodes, self.use_one)
            sampler_3 = Categorical(prob_3)
            index_3 = sampler_3.sample()
            node_3 = candidate_actions_3[index_3]
            selected_nodes.append(node_3)
            log_prob = log_prob + sampler_3.log_prob(index_3)

            
        else:
            edges = list(state.edges())
            prob_1 =  self.policy(x, adj_tensor, edges, None,self.use_one)
            sampler_1 = Categorical(prob_1)
            index_1 = sampler_1.sample()
            len_edges = len(edges)
            if index_1>= len_edges:
                edge_index = index_1 - len_edges
            else:
                edge_index = index_1
            
            selected_edge = edges[edge_index]

            
            selected_nodes = [selected_edge[0], selected_edge[1]]

            log_prob = sampler_1.log_prob(index_1)

            
            if self.edge_combine != 'concat':

                prob_2 = self.policy(x, adj_tensor, None, selected_nodes, self.use_one)
                sampler_2 = Categorical(prob_2)
                index_2 = sampler_2.sample()
                node_1 = selected_nodes[index_2]
                node_2 = selected_nodes[1-index_2]
                selected_nodes = [node_1, node_2]

                log_prob = log_prob + sampler_2.log_prob(index_2)

                
            
            candidate_actions_3 = self.env.constraint_action_space(selected_nodes)
            if len(candidate_actions_3) == 0:
                return None, None

            prob_3 = self.policy(x, adj_tensor, candidate_actions_3, selected_nodes, self.use_one)
            sampler_3 = Categorical(prob_3)
            index_3 = sampler_3.sample()
            node_3 = candidate_actions_3[index_3]
            selected_nodes.append(node_3)
            log_prob = log_prob + sampler_3.log_prob(index_3)

        return selected_nodes, log_prob

class RandomAgent():
    def __init__(self,env, edge_based=0):
        self.env = env
        self.edge_based = edge_based

    def select_action(self,state):
        if not self.edge_based:
            num_nodes = nx.number_of_nodes(state)
            candidate_actions_1 = list(range(num_nodes))
            node_1 = random.choice(candidate_actions_1)
            selected_nodes = [node_1]

            candidate_actions_2 = self.env.constraint_action_space(selected_nodes)
            if len(candidate_actions_2) == 0:
                return None
            node_2 = random.choice(candidate_actions_2)
            selected_nodes.append(node_2)

            candidate_actions_3 = self.env.constraint_action_space(selected_nodes) 
            if len(candidate_actions_3) == 0:
                return None
            node_3 = random.choice(candidate_actions_3)
            selected_nodes.append(node_3)  
        else:
            edges = list(state.edges())
            selected_edge = random.choice(edges)
            
            first_index = random.choice([0,1])
            second_index = 1 - first_index
            selected_nodes = [selected_edge[first_index],selected_edge[second_index]]
            
            candidate_actions_3 = self.env.constraint_action_space(selected_nodes) 
            if len(candidate_actions_3) == 0:
                return None
            node_3 = random.choice(candidate_actions_3)
            selected_nodes.append(node_3)  
            
            
            

        return selected_nodes  














def update_policy(gamma, optimizer, reward_episode, policy_history):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    R=0
    rewards = []
    for r in reward_episode[::-1]:
        R = r + gamma*R
        rewards.insert(0,R)

    rewards = torch.FloatTensor(rewards).to(device)

    loss = torch.sum(torch.mul(policy_history.view(-1), Variable(rewards)).mul(-1), -1).to(device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






def train_multiple_graphs(bmname,num_possible_actions, include_self, new_epoch, negative_reward, nor, features, f, action_percent=None, adaptive_reward=0, use_one = 0, use_graph_state=1, md =1, learning_rate=0.001, model_epoch=35, use_all=0, args=None, embedding_normalize=False, padding=False, edge_based = 0, edge_combine = 'mult', num_hidden_layers=0, train_ratio = 0.8, attack_model_save_dir = None):

    model_input_dim = 10
    model_hidden_dim = 64
    method = 'base'

    feature = features
    epoch = model_epoch
    nc = 3
    bn = 'False'

    savedir = 'model_data/'+ bmname + '/' + 'ip_dim_' +  str(model_input_dim) + '_hd_dim_' + str(model_hidden_dim) + '_nc_' + str(nc)   + '_bn_' + str(bn)   + '/'
    model_dir = savedir + 'method_'+ str(method) + '_f_' + str(feature) +  '_epoch_' + str(epoch)  + 'model.p'
    
    input_dim = 10
    
    if f == 'struct':
        input_dim = 2
    elif f == 'struct2':
        input_dim = 22
        
    hidden_dim = 64
    embedding_dim = 64
    num_layers = 2
    pred_hidden = 64
    gamma = 0.99

#     learning_rate = 0.001
    val_graph_dir = savedir + 'val_graphs.p'
    with open(val_graph_dir, 'rb') as f2:
        graph_list = pickle.load(f2)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    length_graph_list = len(graph_list)

    train_graph_length = int(length_graph_list*train_ratio)
    train_graph_list = graph_list[0:train_graph_length]
    test_graph_list = graph_list[train_graph_length:length_graph_list]
    num_episodes = new_epoch*train_graph_length
    

    classifier = torch.load(model_dir)
    print('loaded model from: ', model_dir)
    classifier.eval()
    env = GraphEnv(classifier, include_self, feature, use_all)
    agent = Agent(input_dim, hidden_dim, embedding_dim, num_layers,pred_hidden, env, nor, use_one, use_graph_state, md,args, embedding_normalize, padding, edge_based, edge_combine, num_hidden_layers)
    optimizer = optim.Adam(agent.policy.parameters(), lr = learning_rate)

    episode_length = []


    for episode in range(num_episodes):

        graph_index = episode%train_graph_length


        reward_episode = []
        policy_history = Variable(torch.Tensor().to(device))
        policy_history_list = []
        
        if action_percent is not None:
            num_possible_actions = int(train_graph_list[graph_index].number_of_edges()*action_percent)
            if num_possible_actions == 0:
                num_possible_actions = num_possible_actions + 1  
                
        if adaptive_reward:
            negative_reward = - (1/num_possible_actions)
            
                
        
        state, ori_label =  agent.env.reset(train_graph_list[graph_index],negative_reward)


        
        if f == 'struct':
            x_ = np.zeros((train_graph_list[graph_index].number_of_nodes(),input_dim), dtype = float)
            for i,u in enumerate(state.nodes()):
                deg = state.degree(u)
                clustering = nx.clustering(state,u)
                x_[i,0] = deg
                x_[i,1] = clustering
                x = torch.from_numpy(x_).type(torch.FloatTensor).to(device)

        elif f == 'struct2':
            max_deg = 10
            adj = np.array(nx.to_numpy_matrix(state))
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs>10] = 10
            feat = np.zeros((len(degs), max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1 
            
            clusterings = np.array(list(nx.clustering(state).values()))
            clusterings = np.expand_dims(clusterings,axis=1)
            x_ = np.hstack([feat, clusterings])
            x = torch.from_numpy(x_).type(torch.FloatTensor).to(device)
            y = torch.ones(state.number_of_nodes(),10).to(device)
            x = torch.cat((x,y),1)
        else:
            x = torch.ones(state.number_of_nodes(),input_dim).to(device)
            
        if padding:
            x = pad(x, input_dim, device)
            
            
        

        for i in range(num_possible_actions):

            action, log_prob = agent.select_action(x, state)

            if action is not None:

                policy_history_list.append(log_prob.view(1,1))



                state,  reward, done , current_label = agent.env.step(action)

                policy_history = torch.cat([*policy_history_list])

                reward_episode.append(reward)
                if done:
                    break


        episode_length.append(len(reward_episode))

        update_policy(gamma, optimizer, reward_episode, policy_history)

    

    

    print('In the training set: ')
    episode_length_train = []
    train_succ_len = []
    train_correct_label = []
    train_original_predicted_label = []
    train_after_attack_label = []
    
    train_succ_01 = []
    train_succ_02 = []
    train_succ_03 = []
    train_succ_04 = []
    
    
    sucecess_train = [0]*len(train_graph_list)
    for i in range(len(train_graph_list)):
        if action_percent is not None:
            num_possible_actions = int(train_graph_list[i].number_of_edges()*action_percent)
            if num_possible_actions == 0:
                num_possible_actions = num_possible_actions + 1
                
            num_possible_actions_01 =  int(train_graph_list[i].number_of_edges()*0.01)
            if num_possible_actions_01 == 0:
                num_possible_actions_01 = 1
            num_possible_actions_02 =  int(train_graph_list[i].number_of_edges()*0.02) 
            if num_possible_actions_02 == 0:
                num_possible_actions_02 = 1
            num_possible_actions_03 =  int(train_graph_list[i].number_of_edges()*0.03) 
            if num_possible_actions_03 == 0:
                num_possible_actions_03 = 1
            num_possible_actions_04 =  int(train_graph_list[i].number_of_edges()*0.04) 
            if num_possible_actions_04 == 0:
                num_possible_actions_04 = 1
        if adaptive_reward:
            negative_reward = -(1/num_possible_actions)
        state, ori_label = agent.env.reset(train_graph_list[i],negative_reward)
        
        
        
        


        if f == 'struct':
            x_ = np.zeros((state.number_of_nodes(),input_dim), dtype = float)
            for j,u in enumerate(state.nodes()):
                deg = state.degree(u)
                clustering = nx.clustering(state,u)
                x_[j,0] = deg
                x_[j,1] = clustering
                x = torch.from_numpy(x_).type(torch.FloatTensor).to(device)
               
        elif f == 'struct2':
            max_deg = 10
            adj = np.array(nx.to_numpy_matrix(state))
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs>10] = 10
            feat = np.zeros((len(degs), max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1 
            
            clusterings = np.array(list(nx.clustering(state).values()))
            clusterings = np.expand_dims(clusterings,axis=1)

            x_ = np.hstack([feat, clusterings])
            x = torch.from_numpy(x_).type(torch.FloatTensor).to(device)
            y = torch.ones(state.number_of_nodes(),10).to(device)
            x = torch.cat((x,y),1)
        else:
            x = torch.ones(state.number_of_nodes(),input_dim).to(device)
            
        if padding:
            x = pad(x, input_dim, device)

        

        for j in range(num_possible_actions):
            action, log_prob = agent.select_action(x, state)

            if action is not None:
                state, reward, done, current_label = agent.env.step(action)
                if done:

                    sucecess_train[i] = 1
                    train_succ_len.append(j+1)
                    if (j+1) <= num_possible_actions_01:
                        train_succ_01.append(j+1)
                    if (j+1) <= num_possible_actions_02:
                        train_succ_02.append(j+1)
                    if (j+1) <= num_possible_actions_03:
                        train_succ_03.append(j+1)   
                    if (j+1) <= num_possible_actions_04:
                        train_succ_04.append(j+1) 
                    break
        train_after_attack_label.append(current_label)
        episode_length_train.append(j+1)

    
    print('succ rate: ',sucecess_train.count(1)/len(sucecess_train))
    print('Train suc -rate 0.01', len(train_succ_01)/len(sucecess_train))
    print('Train suc -rate 0.02', len(train_succ_02)/len(sucecess_train)) 
    print('Train suc -rate 0.03', len(train_succ_03)/len(sucecess_train))
    print('Train suc -rate 0.04', len(train_succ_04)/len(sucecess_train))
    
    

    



                                              
                                              
    

    print('testing: ')
    episode_length_test = []
    test_succ_len = []
    test_correct_label = []
    test_original_predicted_label = []
    test_succ_01 = []
    test_succ_02 = []
    test_succ_03 = []
    test_succ_04 = []
                                              
    test_after_attack_label = []                                             
    sucecess = [0]*len(test_graph_list)
    for i in range(len(test_graph_list)):
        if action_percent is not None:
            num_possible_actions = int(test_graph_list[i].number_of_edges()*action_percent)
            if num_possible_actions == 0:
                num_possible_actions = num_possible_actions + 1
            num_possible_actions_01 =  int(test_graph_list[i].number_of_edges()*0.01)
            if num_possible_actions_01 == 0:
                num_possible_actions_01 = 1
            num_possible_actions_02 =  int(test_graph_list[i].number_of_edges()*0.02) 
            if num_possible_actions_02 == 0:
                num_possible_actions_02 = 1
            num_possible_actions_03 =  int(test_graph_list[i].number_of_edges()*0.03) 
            if num_possible_actions_03 == 0:
                num_possible_actions_03 = 1
            num_possible_actions_04 =  int(test_graph_list[i].number_of_edges()*0.04) 
            if num_possible_actions_04 == 0:
                num_possible_actions_04 = 1
            
        if adaptive_reward:
            negative_reward = -(1/num_possible_actions)
        state, ori_label = agent.env.reset(test_graph_list[i],negative_reward)
        test_correct_label.append(state.graph['label'])
        test_original_predicted_label.append(ori_label)
        
        if f == 'struct':
            x_ = np.zeros((state.number_of_nodes(),input_dim), dtype = float)
            for j,u in enumerate(state.nodes()):
                deg = state.degree(u)
                clustering = nx.clustering(state,u)
                x_[j,0] = deg
                x_[j,1] = clustering
                x = torch.from_numpy(x_).type(torch.FloatTensor).to(device)

        elif f == 'struct2':
            max_deg = 10
            adj = np.array(nx.to_numpy_matrix(state))
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs>10] = 10
            feat = np.zeros((len(degs), max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1 
            
            clusterings = np.array(list(nx.clustering(state).values()))
            clusterings = np.expand_dims(clusterings,axis=1)

            x_ = np.hstack([feat, clusterings])
            x = torch.from_numpy(x_).type(torch.FloatTensor).to(device)
            y = torch.ones(state.number_of_nodes(),10).to(device)
            x = torch.cat((x,y),1)
                
        else:
            x = torch.ones(state.number_of_nodes(),input_dim).to(device)
            
        if padding:
            x = pad(x, input_dim, device)


        for j in range(num_possible_actions):
            action, log_prob = agent.select_action(x, state)
            if action is not None:
                # j = j+1
                state, reward, done, current_label = agent.env.step(action)
                if done:

                    sucecess[i] = 1
                    test_succ_len.append(j+1)
                    if (j+1) <= num_possible_actions_01:
                        test_succ_01.append(j+1)
                    if (j+1) <= num_possible_actions_02:
                        test_succ_02.append(j+1)
                    if (j+1) <= num_possible_actions_03:
                        test_succ_03.append(j+1)
                    if (j+1) <= num_possible_actions_04:
                        test_succ_04.append(j+1)
                    break
        test_after_attack_label.append(current_label)             
        episode_length_test.append(j+1)
  
        
    
    
    print('succ rate: ',sucecess.count(1)/len(sucecess))
    print('Test suc -rate 0.01', len(test_succ_01)/len(sucecess))
    print('Test suc -rate 0.02', len(test_succ_02)/len(sucecess)) 
    print('Test suc -rate 0.03', len(test_succ_03)/len(sucecess))
    print('Test suc -rate 0.04', len(test_succ_04)/len(sucecess)) 



    if attack_model_save_dir is not None:
        torch.save(agent,attack_model_save_dir)
        print('attack_model_saved at: ', attack_model_save_dir)

def pad(x, input_dim, device):
    pad_zeros = torch.zeros(1,input_dim).to(device)
    x = torch.cat((x,pad_zeros),0)
    return x
    





def ave(list):
    print('Length: ',len(list))
    ave = 0
    for i in list:
        ave = ave + i/len(list)
        
    print('Ave: ', ave)


    
    
def random_policy_multigraphs(bmname, num_possible_actions, include_self, features, action_percent,model_epoch=35,use_all=0, edge_based=0, train_ratio = 0.8):
    model_input_dim = 10
    model_hidden_dim = 64
    method = 'base'

    feature = features
    epoch = 35
    nc = 3
    bn = 'False'

    savedir = 'model_data/'+ bmname + '/' + 'ip_dim_' +  str(model_input_dim) + '_hd_dim_' + str(model_hidden_dim) + '_nc_' + str(nc)   + '_bn_' + str(bn)   + '/'
    model_dir = savedir + 'method_'+ str(method) + '_f_' + str(feature) +  '_epoch_' + str(epoch)  + 'model.p'
    
    input_dim = 10
    hidden_dim = 64
    embedding_dim = 64
    num_layers = 2
    pred_hidden = 64
    gamma = 0.99
    num_episodes = 1000
    learning_rate = 0.001
    val_graph_dir = savedir + 'val_graphs.p'
    with open(val_graph_dir, 'rb') as f2:
        graph_list = pickle.load(f2)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    length_graph_list = len(graph_list)

    train_graph_length = int(length_graph_list*train_ratio)
    train_graph_list = graph_list[0:train_graph_length]
    test_graph_list = graph_list[train_graph_length:length_graph_list]
    
    classifier = torch.load(model_dir)
    print('Loaded model from: ', model_dir)
    classifier.eval()
    env = GraphEnv(classifier, include_self, feature, use_all)
                                              
    
    train_suc = [0]*train_graph_length
    test_suc = [0]*len(test_graph_list)
    
    episode_length = []
    agent = RandomAgent(env, edge_based)
    train_succ_len = []
    train_correct_label = []
    train_original_predicted_label = []
    train_after_attack_label = []
    
    train_succ_01 = []
    train_succ_02 = []
    train_succ_03 = []

    for i in range(train_graph_length):
        reward_episode = []
        state, ori_label =  agent.env.reset(train_graph_list[i],-0.5)
        train_correct_label.append(state.graph['label'])
        train_original_predicted_label.append(ori_label)

        if action_percent is not None:
            num_possible_actions = int(train_graph_list[i].number_of_edges()*action_percent)
            if num_possible_actions == 0:
                num_possible_actions = num_possible_actions + 1  
                
            num_possible_actions_01 =  int(train_graph_list[i].number_of_edges()*0.01)
            if num_possible_actions_01 == 0:
                num_possible_actions_01 = 1
            num_possible_actions_02 =  int(train_graph_list[i].number_of_edges()*0.02) 
            if num_possible_actions_02 == 0:
                num_possible_actions_02 = 1
            num_possible_actions_03 =  int(train_graph_list[i].number_of_edges()*0.03) 
            if num_possible_actions_03 == 0:
                num_possible_actions_03 = 1
                
                

        for j in range(num_possible_actions):
            action = agent.select_action(state)
            if action is not None:
                _,reward, done, current_label = agent.env.step(action)
                reward_episode.append(reward)
                if done:
                    train_suc[i] = 1
                    train_succ_len.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_01:
                        train_succ_01.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_02:
                        train_succ_02.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_03:
                        train_succ_03.append(len(reward_episode))
                    
                    break
        train_after_attack_label.append(current_label)
                                              

        episode_length.append(len(reward_episode))

    ave(episode_length)
    print('-------suc_len_ave: ')
    ave(train_succ_len)
    print('train_suc: ', train_suc)
    print('train_suc_rate: ', train_suc.count(1)/len(train_suc))
    
    print('Train suc -rate 0.01', len(train_succ_01)/len(train_suc))
    print('Train suc -rate 0.02', len(train_succ_02)/len(train_suc)) 
    print('Train suc -rate 0.03', len(train_succ_03)/len(train_suc))

    
    test_episode_length = []
    test_succ_len = []
    test_correct_label = []
    test_original_predicted_label = []
    
    test_succ_01 = []
    test_succ_02 = []
    test_succ_03 = []
    test_succ_04 = []
                                              
    test_after_attack_label = [] 
    for i in range(len(test_graph_list)):
        reward_episode = []
        state, test_ori_label =  agent.env.reset(test_graph_list[i],-0.5)
        test_correct_label.append(state.graph['label'])
        test_original_predicted_label.append(test_ori_label)
        if action_percent is not None:
            num_possible_actions = int(test_graph_list[i].number_of_edges()*action_percent)
            if num_possible_actions == 0:
                num_possible_actions = num_possible_actions + 1  

            num_possible_actions_01 =  int(test_graph_list[i].number_of_edges()*0.01)
            if num_possible_actions_01 == 0:
                num_possible_actions_01 = 1
            num_possible_actions_02 =  int(test_graph_list[i].number_of_edges()*0.02) 
            if num_possible_actions_02 == 0:
                num_possible_actions_02 = 1
            num_possible_actions_03 =  int(test_graph_list[i].number_of_edges()*0.03) 
            if num_possible_actions_03 == 0:
                num_possible_actions_03 = 1
            num_possible_actions_04 =  int(test_graph_list[i].number_of_edges()*0.04) 
            if num_possible_actions_04 == 0:
                num_possible_actions_04 = 1
                
        

        for j in range(num_possible_actions):
            action = agent.select_action(state)
            if action is not None:
                # j = j+1
                _,reward, done, current_label = agent.env.step(action)
                reward_episode.append(reward)
                if done:
                    test_suc[i] = 1
                    test_succ_len.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_01:
                        test_succ_01.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_02:
                        test_succ_02.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_03:
                        test_succ_03.append(len(reward_episode))
                    if len(reward_episode)<= num_possible_actions_04:
                        test_succ_04.append(len(reward_episode))
                    break
        test_after_attack_label.append(current_label)

        test_episode_length.append(len(reward_episode))

    print('test_suc: ', test_suc)
    print('test_suc_rate: ', test_suc.count(1)/len(test_suc))
    print('Test suc -rate 0.01', len(test_succ_01)/len(test_suc))

    print('Test suc -rate 0.02', len(test_succ_02)/len(test_suc)) 

    print('Test suc -rate 0.03', len(test_succ_03)/len(test_suc))

    print('Test suc -rate 0.04', len(test_succ_04)/len(test_suc))


    
    
    
    
    
    
    
    































def get_args():
    parser = argparse.ArgumentParser(description = 'Show description')

    parser.add_argument('-data', '--dataset', type = str,
                        help = 'which dataset to run', default = 'REDDIT-MULTI-12K')
    parser.add_argument('-inc_s', '--include_self', type = int,
                        help = 'whether include_self in the candidates', default = 1)
    parser.add_argument('-npa', '--num_possible_actions', type = int,
                        help = 'number of possible actions', default = 4)
    parser.add_argument('-new_ep', '--new_epoch', type = int,
                        help = 'train epoch for rl', default = 2)
    parser.add_argument('-neg_r', '--negative_reward', type = float,
                        help = 'negative reward', default = -0.5)
    parser.add_argument('-nor', '--nor', type = int,
                        help = 'whether normalize adj', default = 1)
    parser.add_argument('-tf', '--train_features', type = str,
                        help = 'train features used for the classifier model', default = 'default')
    parser.add_argument('-me', '--model_epoch', type = int,
                        help = 'model epoch for the classifier model', default = 35)
    parser.add_argument('-f', '--f', type = str,
                        help = 'features used for the attack model', default = 'default')
    parser.add_argument('-uso', '--use_one', type = int,
                        help = 'use_one_graph_encoder', default = 1)
    parser.add_argument('-lr', '--learning_rate', type = float,
                        help = 'learning rate', default = 0.001)
    parser.add_argument('-acp', '--action_percent', type = float,
                        help = 'action_percent', default = 0.03)
    parser.add_argument('-adr', '--adaptive_reward', type = int,
                        help = 'adaptive_reward', default = 1)
    parser.add_argument('-usg', '--use_graph_state', type = int,
                        help = 'use_graph_state', default = 1)
    parser.add_argument('-usa', '--use_all', type = int,
                        help = 'use_all_nodes as candidate', default = 0)
    parser.add_argument('-ebn', '--embedding_normalize', type = int,
                        help = 'normalize embedding', default = 0)
    parser.add_argument('-bias', '--bias', type = int,
                        help = 'use bias or not', default = 1)
    parser.add_argument('-padding', '--padding', type = int,
                        help = 'padding or not', default = 1)
    parser.add_argument('-egb', '--edge_based', type = int,
                        help = 'use_edge_based methods', default = 1)
    parser.add_argument('-egc', '--edge_combine', type = str,
                        help = 'method to get edge_embedding', default = 'mult')
    parser.add_argument('-nhl', '--num_hidden_layers', type = int,
                        help = 'num_hidden_layers', default = 0)
    parser.add_argument('-tr', '--train_ratio', type = float,
                        help = 'train_ratio', default = 0.8)
    parser.add_argument('-seed', '--torch_seed', type = int,
                        help = 'torch_seed', default = 11)
    parser.add_argument('-random', '--random', type = int,
                        help = 'run_random', default = 0)
    
    
    
    

    
    args = parser.parse_args()
    return args
































        













def main():
    args = get_args()
    print(args)
    torch_seed = args.torch_seed
    random.seed(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    bmname = args.dataset
    graph_index = 1
    num_possible_actions = args.num_possible_actions
    include_self = args.include_self
    new_epoch = args.new_epoch
    negative_reward = args.negative_reward
    nor = args.nor
    train_features = args.train_features
    
    
    f = args.f
    use_one = args.use_one
    md =0
    learning_rate = args.learning_rate
    
    action_percent = args.action_percent
    adaptive_reward = args.adaptive_reward
    use_graph_state=args.use_graph_state
    model_epoch = args.model_epoch
    use_all = args.use_all
    embedding_normalize = args.embedding_normalize
    bias = args.bias
    
    padding = args.padding
    edge_based = args.edge_based
    edge_combine = args.edge_combine
    num_hidden_layers  = args.num_hidden_layers
    train_ratio = args.train_ratio
    
    attack_model_save_path ='attack_models/'+ 'bm_' + bmname + '_mtf_' + train_features + '_atf_' + f + 'md_epoch' + str(model_epoch) + '/'
    if not os.path.exists(attack_model_save_path):
        os.makedirs(attack_model_save_path)
    
    model_name = 'tr_' + str(train_ratio) + '_use_one' + str(use_one) + '_lr' + str(learning_rate) + '_acp_' + str(action_percent) + '_atr_' + str(adaptive_reward) + '_usg_' + str(use_graph_state) + '_useall_' + str(use_all) + '_emn_' +str(embedding_normalize) + 'bias' + str(bias) + '_pdi_' +str(padding) + '_egb_' +str(edge_based) + '_egc_' + str(edge_combine) + '_nhl_' + str(num_hidden_layers) + '.p' 
    attack_model_save_dir = attack_model_save_path + model_name
    
    
    
    
    if not args.random:
        train_multiple_graphs(bmname,num_possible_actions, include_self, new_epoch, negative_reward, nor, train_features, f, action_percent, adaptive_reward, use_one, use_graph_state, md, learning_rate, model_epoch, use_all, args, embedding_normalize, padding, edge_based, edge_combine, num_hidden_layers, train_ratio, attack_model_save_dir)
    else:
        random_policy_multigraphs(bmname,num_possible_actions, include_self, train_features, action_percent, model_epoch,use_all, edge_based, train_ratio)







if __name__ == '__main__':
    main()

































