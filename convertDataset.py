import numpy as np
import networkx as nx


import os
import os.path as osp
import urllib.request


urls =  {'train':"https://bnn.upc.edu/download/ch21-training-dataset",
    'val': "https://bnn.upc.edu/download/ch21-validation-dataset",
    'test':"https://bnn.upc.edu/download/ch21-test-dataset"
    }

def download_dataset():
    os.makedirs('./dataset',exist_ok=True)
    for k,v in urls.items():
       urllib.request.urlretrieve(v, f'./dataset/{k}.tar.gz')

def extract_tarfiles():
    import tarfile
    for k,v in urls.items():
        tar = tarfile.open( f'./dataset/{k}.tar.gz')
        tar.extractall('./dataset')
        tar.close()


def generator(data_dir, shuffle=False):
    tool = DatanetAPI(data_dir.decode('UTF-8'), shuffle=shuffle)
    it = iter(tool)
    num_samples = 0
    for sample in it:
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        P = sample.get_port_stats()
        HG = network_to_hypergraph(network_graph=G_copy,
                                   routing_matrix=R,
                                   traffic_matrix=T,
                                   performance_matrix=D,
                                   port_stats=P)
        num_samples += 1
        yield hypergraph_to_input_data(HG)

def network_to_hypergraph(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats):
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    P = port_stats
    #EDGE TYPES: 0 - path to link; 1 - path to node; 2- link to node

    D_G = nx.DiGraph()
    
    for src in range(G.number_of_nodes()):
        D_G.add_node('n_{}'.format(src),**dict((f'n_{k}',v) for k,v in G.nodes[src].items()))
    
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:

                if G.has_edge(src, dst):
                    #Create node corresponding to edge
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 l_capacity=G.edges[src, dst]['bandwidth'],
                                 out_occupancy=P[src][dst]['qosQueuesStats'][0]['avgPortOccupancy'] /
                                            G.nodes[src]['queueSizes'])
                    D_G.add_edge('l_{}_{}'.format(src, dst),'n_{}'.format(src),edge_type=2)
                    D_G.add_edge('l_{}_{}'.format(src, dst),'n_{}'.format(dst),edge_type=2)
                    
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        dct_flows = dict((f'p_{k}',v) for k,v in T[src, dst]['Flows'][f_id].items())
                        dct_flows.pop('p_SizeDistParams')
                        dct_flows.pop('p_TimeDistParams')
                        dct_flows_size = dict((f'p_size_{k}',v) for k,v in T[src, dst]['Flows'][f_id]['SizeDistParams'].items())
                        dct_flows_time = dict((f'p_time_{k}',v) for k,v in T[src, dst]['Flows'][f_id]['TimeDistParams'].items())
                        dct_flows.update(dct_flows_size)
                        dct_flows.update(dct_flows_time)
                        dct_flows['out_delay'] = D[src, dst]['Flows'][f_id]['AvgDelay']
                        
                        D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),**dct_flows)

                        for j, (h_1, h_2) in enumerate([R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]):
                            _p = 'p_{}_{}_{}'.format(src, dst, f_id)
                            _l =  'l_{}_{}'.format(h_1, h_2)
                            _n1 =  'n_{}'.format(h_1)
                            _n2 =  'n_{}'.format(h_2)
                            if _n1 not in D_G[_p]:
                               D_G.add_edge(_p,_n1,edge_type=1)
                            if _n2 not in D_G[_p]:
                               D_G.add_edge(_p,_n2,edge_type=1)
                            D_G.add_edge(_p,_l,edge_type=0)
                            
                            
    D_G.remove_nodes_from([node for node, out_degree in D_G.	degree() if out_degree == 0])

    return D_G


"""
    Teste de performance 

"""
import torch_geometric
import torch
def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
    
    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            L= data.get(str(key),None)
            if L is None:
                data[key] = [value]
            else:
                L.append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            L= data.get(str(key),None)
            if L is None:
                data[key] = [value]
            else:
                L.append(value)

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data

total_samples = {'train':120000 ,
        'validation':3120,
         'test':1560
        }
import datanetAPI
from tqdm import tqdm, trange
from torch_geometric.data import InMemoryDataset,Dataset, Data,  DataLoader, download_url, extract_zip

def process_file(file_num,mode='validation'):
    os.makedirs(f'./dataset/converted_{mode}',exist_ok=True)

    reader = datanetAPI.DatanetAPI(f'./dataset/gnnet-ch21-dataset-{mode}',
                                  intensity_values=[],topology_sizes=[],shuffle=False)
    reader._selected_tuple_files = [reader._all_tuple_files[file_num]]
    print(reader._selected_tuple_files)
    for i,sample in tqdm(enumerate(iter(reader))):
        G_copy = sample.get_topology_object().copy()


        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        P = sample.get_port_stats()
        HG = network_to_hypergraph(network_graph=G_copy,
                               routing_matrix=R,
                               traffic_matrix=T,
                               performance_matrix=D,
                               port_stats=P)

        data = from_networkx(HG)
        data.edge_index = data.edge_index.int()

        def name_to_id(s):
            s = s[0]
            if s == 'p':
                return 0
            elif s == 'l':
                return 1
            elif s == 'n':
                return 2
            raise Exception("node does not begin with p,l or n ")

        data.type =  torch.as_tensor(np.array([name_to_id(name) for name in HG.nodes]))
        data.g_delay =   sample.get_global_delay()
        data.g_losses =   sample.get_global_losses()
        data.g_packets =   sample.get_global_packets()
        data.g_AvgPktsLambda = sample.get_maxAvgLambda()
        #data.p_SizeDist = torch.nn.functional.one_hot(data.p_SizeDist,num_classes=4)
        #data.p_TimeDist = torch.nn.functional.one_hot(data.p_TimeDist,num_classes=6)
        #print(data)
        #print(f'Saved to ./dataset/converted_{mode}/{mode}_{file_num}_{i}.pt')
        torch.save(data,f'./dataset/converted_{mode}/{mode}_{file_num}_{i}.pt')

def process_in_parallel(mode,max_proc=8):
    reader = datanetAPI.DatanetAPI(f'./dataset/gnnet-ch21-dataset-{mode}',
                                      intensity_values=[],topology_sizes=[],shuffle=False)
    n_files = len(reader._all_tuple_files)
    import multiprocessing
    pool = multiprocessing.Pool(processes=max_proc) #use all available cores, otherwise specify the number you want as an argument
    for i in range(n_files):
        pool.apply_async(process_file, args=(i,mode))
    pool.close()
    pool.join()
