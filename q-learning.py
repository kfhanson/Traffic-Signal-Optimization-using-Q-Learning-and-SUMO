import os
import sys
import traci
import time
import numpy as np
from collections import defaultdict
import pickle
from sumolib import checkBinary
import traceback

# --- Q-Learning Parameters ---
ALPHA = 0.1        # Learning rate: How much the agent learns from new experiences. A higher value means faster learning.
ALPHA_DECAY_RATE = 0.99  # Decay factor per epoch: Reduces the learning rate over time to stabilize the learning.
MIN_ALPHA = 0.01         # Minimum learning rate: Prevents the learning rate from getting too small.
GAMMA = 0.9              # Discount factor: How much the agent values future rewards over immediate rewards.
NUM_EPOCHS = 50          # Number of passes through the dataset (epochs), controlling how long the agent trains.
Q_TABLE_SAVE_PATH = 'learned_q_table.pkl'  # Path to save the Q-table after training.

# --- Traffic Signal Phase Actions ---
num_actions = 4  # Number of possible actions: 4 traffic signal phases 

# --- State Discretization Bins (Vehicle Counts) ---
vehicle_bins = [5, 15, 30]  # Bins for discretizing the vehicle count into 4 levels: <5, 5-14, 15-29, >=30.
# Each lane has its own discretized value based on the number of halting vehicles.

# --- SUMO Simulation Setup ---
sumo_binary = checkBinary('sumo-gui') 
sumo_config = "osm.sumocfg"  
traffic_light_id = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339' 

# --- Vehicle Count Edges (SUMO network) ---
approach_edges = {
    "north": "754598165#2",  
    "south": "1053267667#3", 
    "east": "749662140#0",   
    "west": "885403818#2",   
}

# --- Q-Table Initialization ---
q_table = defaultdict(lambda: np.zeros(num_actions))  # Initialize the Q-table
current_alpha = ALPHA  # Set initial learning rate.

# --- Discretize Vehicle Count Function ---
def discretize(value, bins):
    """
    Convert a continuous vehicle count to a discrete state using predefined bins.
    Returns the discretized state index based on the count.
    """
    for i, threshold in enumerate(bins):
        if value < threshold:
            return i  
    return len(bins) 

# --- Get State from SUMO Simulation ---
def get_state_from_traci(tls_id):
    """
    Retrieves the current state from the SUMO simulation by getting the number of halting vehicles on each approach.
    Each approach is discretized into one of the bins.
    """
    try:
        north_count = traci.edge.getLastStepHaltingNumber(approach_edges["north"])  # Get halting vehicles count on the north lane.
        south_count = traci.edge.getLastStepHaltingNumber(approach_edges["south"])  # Get halting vehicles count on the south lane.
        east_count = traci.edge.getLastStepHaltingNumber(approach_edges["east"])    # Get halting vehicles count on the east lane.
        west_count = traci.edge.getLastStepHaltingNumber(approach_edges["west"])    # Get halting vehicles count on the west lane.

        # Discretize the counts into states based on predefined bins.
        state_tuple = (
            discretize(north_count, vehicle_bins),
            discretize(south_count, vehicle_bins),
            discretize(east_count, vehicle_bins),
            discretize(west_count, vehicle_bins),
        )
        return state_tuple  # Return the state as a tuple of discretized values.
    except traci.exceptions.TraCIException as e:
        print(f"TraCI error getting state: {e}. Returning None.")
        return None 
    except Exception as e:
        print(f"Unexpected error: {e}. Returning None.")
        traceback.print_exc() 
        return None

# --- Reward Function Based on Average Waiting Time ---
def calculate_reward_from_traci(state_t, state_t_plus_1):
    """
    Reward function that encourages the agent to minimize waiting time across all lanes.
    It also penalizes emergency braking events, which indicate vehicle stoppage due to red lights.
    """
    try:
        # Get the number of halting vehicles on each lane.
        wait_t_north = traci.edge.getLastStepHaltingNumber(approach_edges["north"])
        wait_t_south = traci.edge.getLastStepHaltingNumber(approach_edges["south"])
        wait_t_east = traci.edge.getLastStepHaltingNumber(approach_edges["east"])
        wait_t_west = traci.edge.getLastStepHaltingNumber(approach_edges["west"])

        # Average waiting time across all lanes.
        avg_wait_t = (wait_t_north + wait_t_south + wait_t_east + wait_t_west) / 4.0
        reward = -avg_wait_t  # Negative reward for longer waiting times (to encourage minimizing waiting).

        # Add penalty for emergency braking to avoid vehicles stopping at red lights.
        emergency_braking_count = traci.simulation.getEmergencyBrakingCount()  # Count of emergency braking events.
        reward -= emergency_braking_count * 10  # Heavier penalty for emergency braking.

    except Exception as e:
        print(f"Error calculating reward: {e}")
        reward = 0  # Default reward if there's an error in calculation.
    return reward

# --- Epsilon-Greedy Exploration ---
def epsilon_greedy(state, epsilon=0.1):
    """
    Epsilon-greedy action selection: 
    - With probability epsilon, explore a random action.
    - With probability 1-epsilon, exploit the best-known action based on Q-table values.
    """
    if np.random.rand() < epsilon:
        # Explore: Take a random action.
        return np.random.choice(num_actions)
    else:
        # Exploit: Choose the action with the highest Q-value for the current state.
        return np.argmax(q_table[state])

# --- Main Q-Learning Loop ---
def run_qlearning():
    """
    The main loop where Q-learning is applied across multiple epochs.
    During each epoch, the agent updates its Q-table based on its interactions with the simulation.
    """
    for epoch in range(NUM_EPOCHS):
        epoch_alpha = ALPHA * (ALPHA_DECAY_RATE ** epoch)  # Reduce the learning rate over time.
        current_alpha = max(epoch_alpha, MIN_ALPHA)  # Ensure the learning rate doesn't go below the minimum value.
        print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        for step in range(100):  # Each epoch runs for a fixed number of steps.
            state_t = get_state_from_traci(traffic_light_id)  # Get the current state.

            if state_t is None:  # Skip if state retrieval fails.
                continue

            # Choose an action based on the Q-table using epsilon-greedy strategy.
            chosen_action = epsilon_greedy(state_t, epsilon=0.1)

            # Apply the chosen action by setting the traffic light phase.
            traci.trafficlight.setPhase(traffic_light_id, chosen_action)

            # Step the simulation forward by one step.
            traci.simulationStep()

            # Get the next state after applying the action.
            state_t_plus_1 = get_state_from_traci(traffic_light_id)

            if state_t_plus_1 is None:  # Skip if next state retrieval fails.
                continue

            # Calculate the reward based on the state transitions.
            reward_t = calculate_reward_from_traci(state_t, state_t_plus_1)

            # Update the Q-table using the Q-learning update formula.
            old_value = q_table[state_t][chosen_action]  # Old Q-value for the current state-action pair.
            next_max = np.max(q_table[state_t_plus_1])  # Maximum Q-value for the next state.
            new_value = old_value + current_alpha * (reward_t + GAMMA * next_max - old_value)  # Updated Q-value.
            q_table[state_t][chosen_action] = new_value  # Store the updated Q-value.

            print(f"Step {step}: State: {state_t}, Action: {chosen_action}, Reward: {reward_t}")

        # Save the Q-table after each epoch.
        with open(Q_TABLE_SAVE_PATH, 'wb') as f:
            pickle.dump(dict(q_table), f)  # Save Q-table to a file.
        print(f"Epoch {epoch + 1} completed. Q-table saved.")

# --- Run the SUMO Simulation ---
def run_simulation_executor():
    """
    Start the SUMO simulation and execute the Q-learning algorithm.
    """
    sumo_cmd = [
        sumo_binary, "-c", sumo_config,
        "--tripinfo-output", "tripinfo_q_offline.xml",  # Output trips info for analysis.
        "--summary-output", "summary_q_offline.xml",  # Summary of the simulation output.
        "--waiting-time-memory", "1000",  # Memory to store waiting time data.
        "--duration-log.statistics",  # Log simulation statistics for the entire duration.
        "--time-to-teleport", "-1",  # Disable teleportation (all vehicles follow their routes).
        "--no-step-log", "true"  # Do not log individual steps (for performance).
    ]
    traci.start(sumo_cmd) 
    print("TraCI connection started.")

    try:
        run_qlearning()  
    except Exception as e:
        print(f"Error during Q-learning execution: {e}")
    finally:
        traci.close()  
        print("Simulation finished.")

# --- Main Function ---
if __name__ == "__main__":
    run_simulation_executor()  
