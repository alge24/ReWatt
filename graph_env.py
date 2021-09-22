import networkx
from graph_sampler import GraphSampler
import torch
from torch.autograd import Variable
import copy

class GraphEnv(object):
    def __init__(self, classifier, include_self=0, features = 'default', use_all = 0):
        self.classifier = classifier
        self.classifier.eval()
        self.features = features
        self.reward = 0
        self.done = 0
        self.num_step = 0
        self.negative_reward = -0.5
        self.include_self = include_self
        self.use_all = use_all



    def reset(self, graph, negative_reward):
        self.graph = copy.deepcopy(graph)
        self.label = self.get_predicted_label()

        self.done = 0
        self.negative_reward = negative_reward
        return self.graph, self.label


    def get_predicted_label(self):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        graph_sampler = GraphSampler([self.graph], features = self.features,normalize= False, max_num_nodes = self.graph.number_of_nodes()+1)
        graph_dataloader = torch.utils.data.DataLoader(
            graph_sampler, 
            batch_size=1, 
            shuffle=True,
            num_workers=1)
        for _, b_data in enumerate(graph_dataloader):
            data = b_data
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
        label = Variable(data['label'].long())
        batch_num_nodes = data['num_nodes'].int().numpy() 
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)
        y_pred = self.classifier(h0, adj, batch_num_nodes, assign_x=assign_input)

        _, _label = torch.max(y_pred,1)


        return _label





    def evaluate_reward(self):
        current_label = self.get_predicted_label()
        if current_label == self.label:
            self.reward = self.negative_reward
        else:
            self.reward = 1
            self.done = 1
        return current_label




		







    def step(self, action):

        if self.include_self:
            if action[0]!=action[1]:
                self.graph.remove_edge(action[0],action[1])
        else:
            self.graph.remove_edge(action[0],action[1])

        if self.include_self:
            if action[0]!=action[2]:
                self.graph.add_edge(action[0],action[2])
        else:
            self.graph.add_edge(action[0],action[2])         

        current_label =  self.evaluate_reward()
        self.num_step = self.num_step + 1

        return [self.graph, self.reward, self.done, current_label]

    def constraint_action_space(self, selected_nodes):
        if len(selected_nodes) == 1:
        ### Given the first node
            candidate_nodes = set(self.graph.neighbors(selected_nodes[0]))
            if self.include_self:
                candidate_nodes.add(selected_nodes[0])
            

                
                
        elif len(selected_nodes) == 2:
			### Rewiring, choose a node that is close to the current nodes

            if self.use_all:
                target_node = selected_nodes[0]
                one_hop_neighbors = set(self.graph.neighbors(target_node))
                all_nodes = set(self.graph.nodes)
                candidate_nodes = all_nodes - one_hop_neighbors
                candidate_nodes.add(target_node)
                
                
            else:
            
                target_node = selected_nodes[0]
                two_hop_neighbors = set()
  
                one_hop_neighbors = set(self.graph.neighbors(target_node))
                for i in one_hop_neighbors:
                    i_neighbors = self.graph.neighbors(i)
                    for nei in i_neighbors:
                        two_hop_neighbors.add(nei)
                candidate_nodes = two_hop_neighbors - one_hop_neighbors
                if self.include_self:
                    candidate_nodes.add(target_node)
            

        return list(candidate_nodes)













