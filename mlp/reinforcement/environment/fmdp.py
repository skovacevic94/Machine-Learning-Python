#import numpy as np
import random

import numpy as np

# Finite-State, Finite-Action Markov Decission Process
class FiniteMDP:
    def __init__(self, transitions, rewards, discount_factor):
        self.model = transitions
        self.rewards = rewards
        self.discount_factor = discount_factor

    def next_state(self, curr_state, action):
        next_states_transitions = self.model[curr_state][action]

        states = []
        probs = []
        for next_state, prob in next_states_transitions.items():
            states.append(next_state)
            probs.append(prob)
        sample = np.random.random_sample(1)[0]
        s = 0
        for i in range(len(probs)):
            if sample < s + probs[i]:
                return states[i]
            s = s + probs[i]

    def qvalue_iteration(self, num_iter=100, iteration_callback=None):
        q_value = {}
        v_value = {} # This max_a{q_value(s, a)}. This is used for caching in order to reduce running complexity.
        # Initialize q_values and v_values to 0

        for state, action_dict in self.model.items():
            state_actions_values = {}
            for action in action_dict.keys():
                state_actions_values[action] = 0
            v_value[state] = 0
            q_value[state] = state_actions_values
        
        # Initialize random policy
        #random.seed(a=42)
        if iteration_callback is not None:
            prev_policy = {}
            for state, action_dict in q_value.items():
                prev_policy[state] = random.choice(list(action_dict.keys()))

        for t in range(num_iter):
            prev_q_value = q_value
            for state, action_dict in self.model.items():
                for action, new_state_dict in action_dict.items():
                    expected_v_next = 0
                    for new_state, prob in new_state_dict.items(): #prob is P(state, action, new_state)
                        if new_state is not None: # Not terminal state
                            expected_v_next = expected_v_next + (prob * v_value[new_state])
                    # Update Q-Value by Bellman Equation
                    q_value[state][action] = self.rewards[state] + self.discount_factor*expected_v_next

            # Update v-values for next iteration outside of nested loops above.
            for state, action_dict in q_value.items():
                v_value[state] = action_dict[max(action_dict, key=action_dict.get)]

            if iteration_callback is not None:
                curr_policy = prev_policy
                for state, action_dict in q_value.items():
                    best_action = max(action_dict, key=action_dict.get)
                    if action_dict[best_action] > prev_q_value[state][prev_policy[state]]:
                        curr_policy[state] = best_action
                iteration_callback(t, curr_policy, q_value, v_value)
                prev_policy = curr_policy

        # Compute final optimal policy
        policy = {}
        for state, action_dict in q_value.items():
            policy[state] = max(action_dict, key=action_dict.get)
        return (policy, q_value, v_value)


