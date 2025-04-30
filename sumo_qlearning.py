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

traffic_light_id = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339' # Your TLS ID
Q_TABLE_LOAD_PATH = 'learned_q_table.pkl'
MIN_GREEN_TIME = 10
YELLOW_TIME = 3

avg_wait_bins = [30, 60, 90]

def discretize(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold:
            return i
    return len(bins)

approach_edges = {
    "north": "754598165#2",
    "south": "1053267667#3",
    "east": "749662140#0",
    "west": "885403818#2",
}

def get_state_from_traci(tls_id):
    try:
        north_wait = traci.edge.getWaitingTime(approach_edges["north"])
        south_wait = traci.edge.getWaitingTime(approach_edges["south"])
        east_wait  = traci.edge.getWaitingTime(approach_edges["east"])
        west_wait  = traci.edge.getWaitingTime(approach_edges["west"])

        state_tuple = (
            discretize(north_wait, avg_wait_bins),
            discretize(south_wait, avg_wait_bins),
            discretize(east_wait, avg_wait_bins),
            discretize(west_wait, avg_wait_bins),
        )
        return state_tuple
    except traci.exceptions.TraCIException as e:
        print(f"TraCI error getting state: {e}. Returning None.")
        return None
    except Exception as e:
        print(f"Unexpected error getting state: {e}. Returning None.")
        traceback.print_exc()
        return None

print(f"Loading learned Q-table from {Q_TABLE_LOAD_PATH}...")
num_actions = 2
q_table = None
try:
    with open(Q_TABLE_LOAD_PATH, 'rb') as f:
        q_table_loaded = pickle.load(f)
    if not q_table_loaded:
         print("Warning: Loaded Q-table dictionary is empty.")
         q_table = defaultdict(lambda: np.zeros(num_actions))
    else:
         try:
             first_key = next(iter(q_table_loaded))
             first_q_values = q_table_loaded[first_key]
             if isinstance(first_q_values, np.ndarray):
                 num_actions = first_q_values.shape[0]
                 print(f"Inferred number of actions from loaded Q-table: {num_actions}")
             else:
                 print(f"Warning: Q-values in loaded table might not be NumPy arrays. Attempting len()...")
                 num_actions = len(first_q_values)
                 print(f"Inferred number of actions from loaded Q-table (using len): {num_actions}")
             q_table = defaultdict(lambda: np.zeros(num_actions), q_table_loaded)
             print(f"Q-table loaded successfully with {len(q_table)} states.")
         except StopIteration:
             print("Warning: Loaded Q-table dictionary is empty after loading (StopIteration).")
             q_table = defaultdict(lambda: np.zeros(num_actions))
         except (TypeError, IndexError, AttributeError) as infer_err:
             print(f"Warning: Could not reliably infer num_actions from loaded data ({infer_err}). Using default: {num_actions}")
             q_table = defaultdict(lambda: np.zeros(num_actions), q_table_loaded)
except FileNotFoundError:
    print(f"Error: Q-table file not found at {Q_TABLE_LOAD_PATH}. Cannot execute.")
    exit()
except Exception as e:
    print(f"Error loading or processing Q-table: {e}")
    import traceback
    traceback.print_exc()
    exit()

if q_table is None:
    print("Error: Q-table could not be initialized. Exiting.")
    exit()

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
    print("TraCI connection started.")

    step = 0
    phase_start_time = 0
    num_actions = q_table[list(q_table.keys())[0]].shape[0] if q_table else 2

    action_to_green_phase = {0: 0, 1: 2}
    green_to_yellow_phase = {0: 1, 2: 3}
    try:
        if traffic_light_id not in traci.trafficlight.getIDList():
            print(f"FATAL ERROR: TLS {traffic_light_id} not found in simulation.")
            traci.close(); return

        while traci.simulation.getMinExpectedNumber() > 0:
            current_time = traci.simulation.getTime()
            current_sumo_phase = traci.trafficlight.getPhase(traffic_light_id)

            is_green_phase_active = current_sumo_phase in action_to_green_phase.values()
            min_time_passed = (current_time - phase_start_time) >= MIN_GREEN_TIME

            if (is_green_phase_active and min_time_passed) or not is_green_phase_active:
                state = get_state_from_traci(traffic_light_id)

                if state is not None:
                    q_values = q_table[state]
                    chosen_action = np.argmax(q_values)
                    print(f"Time: {current_time:.1f}, State: {state}, Q-Vals: {q_values}, Chosen Action: {chosen_action}")
                    target_green_phase = action_to_green_phase[chosen_action]

                    if current_sumo_phase != target_green_phase:
                        if current_sumo_phase in green_to_yellow_phase:
                            # Currently green, switch to yellow first
                            yellow_phase = green_to_yellow_phase[current_sumo_phase]
                            traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
                            # Simulate yellow phase duration
                            yellow_end_time = current_time + YELLOW_TIME
                            while traci.simulation.getTime() < yellow_end_time:
                                if traci.simulation.getMinExpectedNumber() == 0: break
                                traci.simulationStep(); step += 1
                            # Set the target green phase *after* yellow
                            if traci.simulation.getMinExpectedNumber() > 0:
                                traci.trafficlight.setPhase(traffic_light_id, target_green_phase)
                                phase_start_time = traci.simulation.getTime() # Reset timer
                        else: # Currently Yellow or Red, switch directly to target green
                            traci.trafficlight.setPhase(traffic_light_id, target_green_phase)
                            phase_start_time = traci.simulation.getTime() # Reset timer

                        if step % 50 == 0:
                           print(f"Time: {current_time:.1f}, State: {state}, Q-Vals: {q_values}, Chosen Action: {chosen_action}, Switching to Phase: {target_green_phase}")

                else:
                    pass

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
        print(f"\nUnexpected Python error during simulation loop at step {step}:")
        traceback.print_exc()
    finally:
        print("Closing TraCI connection.")
        traci.close()
        print("Simulation finished.")


if __name__ == "__main__":
    run_simulation_executor()
    # Analyze the results using your existing function or a similar one
    # analyze_tripinfo("tripinfo_q_offline.xml")
    # print("\nAnalysis of results (tripinfo_q_offline.xml) should be performed.")