import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from datetime import date, datetime

from src.entanglement_management.generation import EntanglementGenerationA
from src.topology.node import QuantumRouter
from src.topology.router_net_topo import RouterNetTopo
from src.app.request_app import RequestApp
import src.utils.log as log

CONFIG_FILE = "config.json"

# meta params
NO_TRIALS = 10
LOGGING = False
LOG_OUTPUT = "example/SC24/exp_1_log.txt"
MODULE_TO_LOG = ["timeline", "memory", "bsm", "generation", "request_app"]


# simulation params
PREP_TIME = int(1e12)  # 1 second
COLLECT_TIME = int(3e12)  # 10 seconds

# qc params
QC_FREQ = 1e11

# application params
APP_NODE_NAME = "left"
OTHER_NODE_NAME = "right"
num_memories = [1,2,4,8,16,32,64]  # iterate through these

# storing data
data_dict = {"Memory Count": [],
             "Average Throughput": [],
             "Std. Throughput": [],
             "Average TTT": [],
             "Std. TTT":[],
             }


collect_time_base = COLLECT_TIME

for i, num_memo in enumerate(num_memories):
    print(f"Running {NO_TRIALS} trials for memory count {num_memo} ({i + 1}/{len(num_memories)})")
    data_dict["Memory Count"].append(num_memo)
    throughputs = np.zeros(NO_TRIALS)
    time_to_thousand = np.zeros(NO_TRIALS)

    COLLECT_TIME = collect_time_base / num_memo

    for trial_no in range(NO_TRIALS):
        # establish network
        net_topo = RouterNetTopo(CONFIG_FILE)

        # timeline setup
        tl = net_topo.get_timeline()
        tl.stop_time = PREP_TIME + COLLECT_TIME

        if LOGGING:
            # set log
            if num_memo == num_memories[0]:
                log.set_logger(__name__, tl, LOG_OUTPUT)
                log.set_logger_level('WARN')
                for module in MODULE_TO_LOG:
                    log.track_module(module)
            elif num_memo == num_memories[1]:
                for module in MODULE_TO_LOG:
                    log.remove_module(module)

        # network configuration
        routers = net_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        bsm_nodes = net_topo.get_nodes_by_type(RouterNetTopo.BSM_NODE)

        # Random seed for performing the simulations
        for j, node in enumerate(routers + bsm_nodes):
            node.set_seed(int(datetime.now().timestamp()) + j + (trial_no * 3))

        # set quantum channel parameters
        for qc in net_topo.get_qchannels():
            qc.frequency = QC_FREQ

        # establish "left" node as the start node.
        start_node = None
        for node in routers:
            if node.name == APP_NODE_NAME:
                start_node = node
                break
        # Checking to see if the start node was established or not
        if not start_node:
            raise ValueError(f"Invalid app node name {APP_NODE_NAME}")

        # Setting the "right" node as the 'end' node
        end_node = None
        for node in routers:
            if node.name == OTHER_NODE_NAME:
                end_node = node
                break
        # Checking to see if the end node was established or not
        if not start_node:
            raise ValueError(f"Invalid other node name {OTHER_NODE_NAME}")

        # Establishing the apps on the start and end nodes.
        app_start = RequestApp(start_node)
        app_end = RequestApp(end_node)

        # initialize and start app
        tl.init()
        app_start.start(OTHER_NODE_NAME, PREP_TIME, PREP_TIME + COLLECT_TIME, num_memo, 1.0)
        tl.run()

        # # Used for debugging
        attempt = app_start.node.total_attempts
        success = app_start.node.succesful_attempts
        ttt = app_start.node.time_to_thousand
        # prob = success/attempt

        print('Attempts: ', attempt)
        print('Success: ', success)
        print('TTT: ', ttt*1e-12)

        throughputs[trial_no] = app_start.get_throughput()
        time_to_thousand[trial_no] = app_start.node.time_to_thousand

        print(f"\tCompleted trial {trial_no + 1}/{NO_TRIALS}")

    print("Finished trials.")

    # Saving the results in a dictionary to later convert into a Pandas data frame to plot.

    avg_throughput = np.mean(throughputs)
    std_throughput = np.std(throughputs)
    avg_TTT = np.mean(time_to_thousand) * 1e-12
    std_TTT = np.std(time_to_thousand) * 1e-12

    # # Used for debugging
    # print(f"Average throughput: {avg_throughput} +/- {std_throughput}")
    # print(f"time to thousand entanglements:", time_to_thousand)

    data_dict["Average Throughput"].append(avg_throughput)
    data_dict["Std. Throughput"].append(std_throughput)
    data_dict["Average TTT"].append(avg_TTT)
    data_dict["Std. TTT"].append(std_TTT)

df = pd.DataFrame(data_dict)