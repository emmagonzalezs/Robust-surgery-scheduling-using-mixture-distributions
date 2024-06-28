import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import itertools


class SurgerySlack:
    def __init__(self):
        self.weights_dictionary = {
            'type1': [0.2, 0.3],
            'type2': [0.5, 0.5]
        }
        self.waiting_list_mixture_surgeries = {
            'patient1': [[], [3.0]],
            'patient2': [[], [2.5]]
        }
        self.waiting_list_individual_surgeries = {
            'patient1': [[(1.0, 0.5), (1.5, 0.7)]],
            'patient2': [[(2.0, 0.4), (0.5, 0.2)]]
        }
        self.overtime_prob = 0.1
        self.slack_stored_alternate = {}

    def delta_expression(self, surgeries_for_slack, delta):
        if isinstance(surgeries_for_slack[0], str):
            surgeries_for_slack = [surgeries_for_slack]

        weights_for_slack = [self.weights_dictionary[surgery[1]] for surgery in surgeries_for_slack]
        patient_id = [patient_info[0] for patient_info in surgeries_for_slack]
        total_mu = sum(self.waiting_list_mixture_surgeries[patient_info[0]][1][0] for patient_info in surgeries_for_slack)
        mu_list, sigma_list = zip(*[zip(*self.waiting_list_individual_surgeries[ids][0]) for ids in patient_id])

        weights = np.array(weights_for_slack, dtype=object)
        mus = np.array(mu_list, dtype=object)
        sigmas = np.array(sigma_list, dtype=object)
        num_elements = [len(sublist) for sublist in weights_for_slack]

        index_combinations = list(itertools.product(*[range(n) for n in num_elements]))
        selected_weights = [[weight[i] for i, weight in zip(indices, weights)] for indices in index_combinations]
        selected_mus = [[mu[i] for i, mu in zip(indices, mus)] for indices in index_combinations]
        selected_sigmas = [[sigma[i] for i, sigma in zip(indices, sigmas)] for indices in index_combinations]
        weight_products = np.prod(np.array(selected_weights), axis=1)
        mu_sums = np.sum(np.array(selected_mus), axis=1)
        sigma_sums = np.sum(np.array(selected_sigmas), axis=1)
        cdf_arguments = (total_mu + delta - mu_sums) / sigma_sums
        cdf_values = norm.cdf(cdf_arguments)
        result = np.sum(weight_products * cdf_values)
        return result

    def to_solve(self, delta, surgeries_for_slack):
        return self.delta_expression(surgeries_for_slack, delta) - (1 - self.overtime_prob)

    def slack_time_alternate(self, surgeries_for_slack):
        if not surgeries_for_slack:
            return 0

        tuple_map = tuple(tuple(sublist) for sublist in surgeries_for_slack)
        if tuple_map in self.slack_stored_alternate:
            return self.slack_stored_alternate[tuple_map]

        solution_slack = fsolve(self.to_solve, x0=np.array([0]), args=surgeries_for_slack)
        if solution_slack[0] < 0:
            print("Slack is negative here")
        self.slack_stored_alternate[tuple_map] = abs(solution_slack[0])
        return abs(solution_slack[0])

    def verify_delta(self, surgeries_for_slack):
        # Get the delta value
        delta = self.slack_time_alternate(surgeries_for_slack)
        # Check if it satisfies the equation
        result = self.delta_expression(surgeries_for_slack, delta)
        expected = 1 - self.overtime_prob
        print(f"Delta: {delta}")
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        print(f"Difference: {abs(result - expected)}")
        return abs(result - expected) < 1e-6  # tolerance level for floating-point comparison

# Example usage
surgery_slack = SurgerySlack()
surgeries_for_slack = [
    ('patient1', 'type1'),
    ('patient2', 'type2')
]
surgery_slack.verify_delta(surgeries_for_slack)
