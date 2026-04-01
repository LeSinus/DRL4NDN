#!/bin/bash

declare -A scenario_map=(
   ["static_with_one_optimal_face"]="state_one simple 3,10,30"
   ["dynamic_RTT"]="state_one,state_two simple 3,10,30"
   ["permanent_faulty_face"]="state_two,state_three simple 3,10,30"
   ["transient_faulty_face"]="state_three simple 3,10,30"
   ["permanent_face_specific"]="state_three,state_four simple 3,10,30"
   ["transient_face_specific"]="state_four simple 3,10,30"
   ["streak_disruption"]="state_four simple,streak 3,10,30"

)


agents=("PPO" "A2C" "DQN" "TRPO" "QRDQN")
# Function to run the agent for a given scenario
run_agent() {
    local scenario=$1
    local states=$2
    local rewards=$3
    local faces=$4

    for reward in $(echo "$rewards" | tr ',' ' '); do
        for state in $(echo "$states" | tr ',' ' '); do
            for face in $(echo "$faces" | tr ',' ' '); do
                for agent in "${agents[@]}"; do
                  echo "Running $agent for reward: $reward, state: $state, scenario: $scenario, face: $face"
                  python3 ../agents/DRL_Agents.py --reward "$reward" --state "$state" --scenario "$scenario" --agent "$agent" --face "$face"
                done
            done
        done
    done
}

# Iterate through scenarios and execute the commands
for scenario in "${!scenario_map[@]}"; do
    states_rewards_faces="${scenario_map[$scenario]}"

    # Extract states, rewards and faces
    states=$(echo "$states_rewards_faces" | awk '{print $1}')
    rewards=$(echo "$states_rewards_faces" | awk '{print $2}')
    faces=$(echo "$states_rewards_faces" | awk '{print $3}')

    run_agent "$scenario" "$states" "$rewards" "$faces"
done
