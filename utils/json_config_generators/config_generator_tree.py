"""This module generates JSON config files for networks in a tree graph configuration.

Help information may also be obtained using the `-h` flag.

Args:
    tree_size (int): number of nodes in the tree.
    branches (int): number of branches per node.
    memo_size (int): number of memories per node.
    qc_length (float): distance between nodes (in km).
    qc_atten (float): quantum channel attenuation (in dB/m).
    cc_delay (float): classical channel delay (in ms).

Optional Args:
    -d --directory (str): name of the output directory (default tmp)
    -o --output (str): name of the output file (default out.json).
    -s --stop (float): simulation stop time (in s) (default infinity).
    -p --parallel: sets simulation as parallel and requires addition args:
        server ip (str): IP address of quantum manager server.
        server port (int): port quantum manager server is attached to.
        num. processes (int): number of processes to use for simulation.
        sync/async (bool): denotes if timelines should be synchronous (true) or not (false).
        lookahead (int): simulation lookahead time for timelines (in ps).
    -n --nodes (str): path to csv file providing process information for nodes.
"""

import argparse
import json
import os

from sequence.utils.config_generator import add_default_args, get_node_csv, generate_node_procs, generate_nodes, generate_classical, final_config, router_name_func
from sequence.topology.topology import Topology
from sequence.topology.router_net_topo import RouterNetTopo



def add_branches(node_names, nodes, index, bsm_names, bsm_nodes, qchannels, cchannels):
    node1 = node_names[index]
    branch_indices = [args.branches*index+i for i in range(1, args.branches+1)]

    for i in branch_indices:
        if i < len(node_names):
            node2 = node_names[i]
            bsm_name = "BSM_{}_{}".format(node1, node2)
            bsm_names.append(bsm_name)
            bsm_node = {Topology.NAME: bsm_name,
                        Topology.TYPE: RouterNetTopo.BSM_NODE,
                        Topology.SEED: i}
            if args.parallel:
                bsm_node[RouterNetTopo.GROUP] = nodes[i][RouterNetTopo.GROUP]
            bsm_nodes.append(bsm_node)

            # qchannels
            qchannels.append({Topology.SRC: node1,
                              Topology.DST: bsm_name,
                              Topology.DISTANCE: args.qc_length * 500,
                              Topology.ATTENUATION: args.qc_atten})
            qchannels.append({Topology.SRC: node2,
                              Topology.DST: bsm_name,
                              Topology.DISTANCE: args.qc_length * 500,
                              Topology.ATTENUATION: args.qc_atten})
            # cchannels
            cchannels.append({Topology.SRC: node1,
                              Topology.DST: bsm_name,
                              Topology.DELAY: args.cc_delay * 1e9})
            cchannels.append({Topology.SRC: node2,
                              Topology.DST: bsm_name,
                              Topology.DELAY: args.cc_delay * 1e9})
            cchannels.append({Topology.SRC: bsm_name,
                              Topology.DST: node1,
                              Topology.DELAY: args.cc_delay * 1e9})
            cchannels.append({Topology.SRC: bsm_name,
                              Topology.DST: node2,
                              Topology.DELAY: args.cc_delay * 1e9})

            bsm_names, bsm_nodes, qchannels, cchannels = \
                add_branches(node_names, nodes, i, bsm_names, bsm_nodes,
                             qchannels, cchannels)
    return bsm_names, bsm_nodes, qchannels, cchannels


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('tree_size', type=int, help='number of nodes in the tree')
parser.add_argument('branches', type=int, help='number of branches per node')
parser = add_default_args(parser)
args = parser.parse_args()

output_dict = {}

# get csv file (if present)
if args.nodes:
    node_procs = get_node_csv(args.nodes)
else:
    node_procs = generate_node_procs(args.parallel, args.tree_size, router_name_func)
router_names = list(node_procs.keys())
nodes = generate_nodes(node_procs, router_names, args.memo_size)

# generate quantum links and bsm connections
bsm_names, bsm_nodes, qchannels, cchannels = \
        add_branches(router_names, nodes, 0, [], [], [], [])
nodes += bsm_nodes
output_dict[Topology.ALL_NODE] = nodes
output_dict[Topology.ALL_Q_CHANNEL] = qchannels

# generate classical links
router_cchannels = generate_classical(router_names, args.cc_delay)
cchannels += router_cchannels
output_dict[Topology.ALL_C_CHANNEL] = cchannels

# write other config options to output dictionary
final_config(output_dict, args)

# write final json
path = os.path.join(args.directory, args.output)
output_file = open(path, 'w')
json.dump(output_dict, output_file, indent=4)

