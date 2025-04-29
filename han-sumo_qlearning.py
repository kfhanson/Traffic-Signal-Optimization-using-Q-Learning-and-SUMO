import os
import sys
import traci
import time
import numpy as np
from collections import defaultdict
import pickle
from sumolib import net, checkBinary
import traceback

try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("SUMO_HOME not found!")
except ImportError as e:
    sys.exit(f"Error importing SUMO tools: {e}")

sumo_binary = checkBinary('sumo-gui') 
sumo_config = "osm.sumocfg"
net_file = "osm.net.xml"

traffic_light_id = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339' 
Q_TABLE_LOAD_PATH = 'learned_q_table.pkl'
MIN_GREEN_TIME = 10  
YELLOW_TIME = 3 

vehicle_bins = [5, 15, 30]  

approach_edges = {
    "north": "754598165#2",
    "south": "1053267667#3",
    "east": "749662140#0",
    "west": "885403818#2",
}

def discretize(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold:
            return i
    return len(bins) 

def load_q_table(path):
    try:
        with open(path, 'rb') as f:
            loaded_q_table = pickle.load(f)
        if not loaded_q_table:
            print("Warning: Loaded Q-table is empty, initializing default Q-table.")
            return defaultdict(lambda: np.zeros(2)) 
        
        first_key = next(iter(loaded_q_table))
        first_q_values = loaded_q_table[first_key]
        num_actions = first_q_values.shape[0] if isinstance(first_q_values, np.ndarray) else len(first_q_values)
        print(f"Successfully loaded Q-table with {len(loaded_q_table)} states and {num_actions} actions.")
        return defaultdict(lambda: np.zeros(num_actions), loaded_q_table)
    except FileNotFoundError:
        print(f"Error: Q-table file not found at {path}. Exiting.")
        exit()
    except Exception as e:
        print(f"Error loading or processing Q-table: {e}")
        traceback.print_exc()
        exit()

q_table = load_q_table(Q_TABLE_LOAD_PATH)

def get_state_from_traci():
    try:
        north_count = traci.edge.getLastStepHaltingNumber(approach_edges["north"])
        south_count = traci.edge.getLastStepHaltingNumber(approach_edges["south"])
        east_count = traci.edge.getLastStepHaltingNumber(approach_edges["east"])
        west_count = traci.edge.getLastStepHaltingNumber(approach_edges["west"])

        state = (
            discretize(north_count, vehicle_bins),
            discretize(south_count, vehicle_bins),
            discretize(east_count, vehicle_bins),
            discretize(west_count, vehicle_bins),
        )
        return state
    except traci.exceptions.TraCIException as e:
        print(f"Error getting state from TraCI: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return None

def run_simulation_executor():
    sumo_cmd = [
        sumo_binary, "-c", sumo_config,
        "--tripinfo-output", "tripinfo_q_offline.xml",
        "--summary-output", "summary_q_offline.xml",
        "--waiting-time-memory", "1000",
        "--duration-log.statistics",
        "--time-to-teleport", "-1",
        "--no-step-log", "true"
    ]
    traci.start(sumo_cmd)
    print("TraCI connection established.")

    step = 0
    phase_start_time = 0
    action_to_green_phase = {0: 0, 1: 2} 
    green_to_yellow_phase = {0: 1, 2: 3} 

    try:
        if traffic_light_id not in traci.trafficlight.getIDList():
            print(f"FATAL ERROR: TLS {traffic_light_id} not found in simulation.")
            traci.close()
            return

        while traci.simulation.getMinExpectedNumber() > 0:  
            current_time = traci.simulation.getTime()
            current_sumo_phase = traci.trafficlight.getPhase(traffic_light_id)

            is_green_phase_active = current_sumo_phase in action_to_green_phase.values()
            min_time_passed = (current_time - phase_start_time) >= MIN_GREEN_TIME

            if (is_green_phase_active and min_time_passed) or not is_green_phase_active:
                state = get_state_from_traci()
                if state:
                    q_values = q_table[state]
                    chosen_action = np.argmax(q_values)
                    target_green_phase = action_to_green_phase[chosen_action]

                    if current_sumo_phase != target_green_phase:
                        if current_sumo_phase in green_to_yellow_phase:
                            yellow_phase = green_to_yellow_phase[current_sumo_phase]
                            traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
                            yellow_end_time = current_time + YELLOW_TIME
                            while traci.simulation.getTime() < yellow_end_time:
                                traci.simulationStep()
                                step += 1
                            traci.trafficlight.setPhase(traffic_light_id, target_green_phase)
                            phase_start_time = traci.simulation.getTime() 
                        else:
                            traci.trafficlight.setPhase(traffic_light_id, target_green_phase)
                            phase_start_time = traci.simulation.getTime()

                        if step % 50 == 0:
                            print(f"Time: {current_time:.1f}, State: {state}, Chosen Action: {chosen_action}, Switching to Green Phase: {target_green_phase}")

            traci.simulationStep()
            step += 1

            if traci.simulation.getMinExpectedNumber() == 0:
                print("Simulation ended: No more vehicles expected.")
                break

    except traci.exceptions.FatalTraCIError as e:
        print(f"\nFatal TraCI Error during simulation: {e}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
    finally:
        print("Closing TraCI connection.")
        traci.close()

if __name__ == "__main__":
    run_simulation_executor()
