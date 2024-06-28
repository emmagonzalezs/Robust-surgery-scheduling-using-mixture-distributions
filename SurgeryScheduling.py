from WaitingList import waiting_list
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
import pickle
import random
from itertools import product


class SurgeryScheduling:
    def __init__(self, number_surgeries, distr, number_of_ors):

        # Initialize waiting list to schedule in ORs
        self.waiting = waiting_list(number_surgeries, distr)
        # print(waiting[0])
        self.waiting_list_mixture_surgeries = self.waiting[0]  # {'general code': [(mu,sigma)]}
        self.waiting_list_individual_surgeries = self.waiting[1]  # {'general code': [(mu11,sigma11), (mu12, sigma12)]}

        # Weights per surgery in waiting list, for delta calculations
        self.weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))  # {"general code": [w11,w12,w13]}
        self.weights_surgeries_waiting_list = [self.weights_dictionary[key] for key in
                                               self.waiting_list_mixture_surgeries if key in self.weights_dictionary]

        self.ors = {i: [] for i in range(1, number_of_ors + 1)}
        self.max_capacity = 300
        self.list_of_ors = [i for i in range(1, number_of_ors + 1)]
        self.overtime_prob = 0.3

    def sort_waiting_list(self):
        expected_durations_waiting = []
        for key, parameters in self.waiting_list_mixture_surgeries.items():
            for t in parameters:
                expected_durations_waiting.append((key, t[0]))

        expected_durations_waiting.sort(reverse=True, key=lambda x: x[1])

        sorted_surgeries = [surgery for surgery, value in expected_durations_waiting]

        return expected_durations_waiting, sorted_surgeries

    # LPT: schedule each job into an OR where current load is the smallest
    def lpt(self):
        sorted_expectations, sorted_ops = self.sort_waiting_list()

        for surgery in sorted_expectations:
            min_or_used = min(self.ors, key=lambda k: sum(x[1] for x in self.ors[k]))
            self.ors[min_or_used].append(surgery)

        return self.ors

    def delta_expression(self, surgeries_in_or, delta):
        # Step 1: Initialize dictionaries
        mixture_params_for_slack = {surgery: self.waiting_list_mixture_surgeries[surgery] for surgery in
                                    surgeries_in_or}
        individual_params_for_slack = {surgery: self.waiting_list_individual_surgeries[surgery] for surgery in
                                       surgeries_in_or}
        weights_for_slack = {surgery: self.weights_dictionary[surgery] for surgery in surgeries_in_or}

        # Step 2: Compute total mu per OR
        total_mu_per_or = sum(sum(t[0] for t in tuples_list) for tuples_list in mixture_params_for_slack.values())

        # Step 3: Generate weight combinations and their products
        num_procedures_per_surgery = [len(weights) for weights in weights_for_slack.values()]
        products_weights = np.array(
            [np.prod([weights_for_slack[surgery][index] for surgery, index in zip(weights_for_slack.keys(), procedure)])
             for procedure in product(*[range(n) for n in num_procedures_per_surgery])])

        # Step 4: Prepare individual parameters as arrays
        individual_params_list = [np.array(individual_params_for_slack[surgery]) for surgery in
                                  individual_params_for_slack]

        # Step 5: Generate combinations of indices for mu and s values
        index_combinations = list(product(*[range(len(params)) for params in individual_params_list]))

        # Step 6: Compute sum of mu and s for each combination
        sum_mu_array = np.array(
            [sum(individual_params_list[i][index][0] for i, index in enumerate(indices)) for indices in
             index_combinations])
        sum_s_array = np.array(
            [sum(individual_params_list[i][index][1] for i, index in enumerate(indices)) for indices in
             index_combinations])

        # Step 7: Compute phi values for each combination
        phi_array = norm.cdf((total_mu_per_or + delta - sum_mu_array) / sum_s_array)

        # Step 8: Calculate the final delta expression using dot product
        delta_expression_value = np.dot(products_weights, phi_array)

        return delta_expression_value

    def to_solve(self, delta, or_index):
        return self.delta_expression(or_index, delta) - (1 - self.overtime_prob)

    def slack_time(self, or_index):
        solution_slack = fsolve(self.to_solve, x0=np.array([0]), args=(or_index, self.overtime_prob))
        print("Solution:", solution_slack[0])
        print("Function value at solution:", self.to_solve(abs(solution_slack[0]), or_index))

        return abs(solution_slack[0])

    def calculate_priority(self, surgeries_slack, surgery_to_add):  # TODO: define priorities for surgeries Z
        current_slack = self.slack_time(surgeries_slack)
        current_surgeries = surgeries_slack
        current_surgeries.append(surgery_to_add)
        new_slack = self.slack_time(current_surgeries)

        delta = new_slack - current_slack
        omega = self.slack_time(surgery_to_add) - delta
########################################################################################################################
    def regret_based_sampling(self, z, samples=1):

        sorted_expectations, sorted_ops = self.sort_waiting_list()
        best_schedule = None
        best_schedule_overtime = float("inf")

        for _ in range(samples):
            current_schedule = {or_index: [] for or_index in self.ors}
            for i in range(0, len(sorted_expectations), z):  # Go through Z=z surgeries each time
                last_surgery = min(i + z, len(sorted_expectations))
                z_surgeries = sorted_expectations[i:last_surgery]

                priorities_z_surgeries = []  # initialize priorities list for z surgeries

                for surgery in z_surgeries:
                    possible_ors = [or_index for or_index in self.ors if sum(t[1] for t in self.ors[or_index]) +
                                    surgery[1] + self.slack_time(
                        or_index) < self.max_capacity]  # assume only one possible OR
                    surgeries_for_slack = self.ors[possible_ors]
                    # for ors, surgeries_ors in self.ors.items():
                    #     total_used = sum(t[1] for t in surgeries_ors)

                    if surgeries_for_slack:  # if any OR where surgery fits, calculate priority and select best OR
                        priority_surgery_i = self.calculate_priority(
                            surgeries_for_slack, surgery)  # TODO: make function for priority calculation (surgery, v_ikt, best or)
                        priorities_z_surgeries.append(priority_surgery_i)

                    else:  # if none without overtime, schedule directly where less overtime occurs
                        min_or_used = min(self.ors.keys(), key=lambda k: sum(t[1] for t in self.ors[k]) - self.max_capacity)
                        self.ors[min_or_used].append(surgery)

                if priorities_z_surgeries:  # non-empy priorities calculated, so some surgeries have to be scheduled
                    for current_or, data in self.ors.items():
                        drawing_probabilities_per_or = draw_probabilities(
                            current_or)  # TODO: make lists of probs per OR
                        # TODO: surgery_priorities should be a list of all the surgeries that can be drawn
                        drawn_surgery = random.choices(surgeries_priorities, drawing_probabilities_per_or, k=1)[0]
                        current_schedule[current_or].append(drawn_surgery)

            total_overtime = sum(
                max(0, sum(surgery[1] for surgery in or_surgeries) - self.max_capacity) for or_surgeries in
                self.ors.values())
            if total_overtime < best_schedule_overtime:
                best_schedule_overtime = total_overtime
                best_schedule = self.ors.copy()  # Make a copy of the current schedule

        return best_schedule, best_schedule_overtime


########################################################################################################################

# scheduler = SurgeryScheduling(number_surgeries=50, distr="normal", number_of_ors=3)


# ors_schedule_lpt = scheduler.lpt()
# total_overtime_lpt = sum(max(0, sum(surgery[1] for surgery in or_surgeries) - scheduler.max_capacity)
#                          for or_surgeries in ors_schedule_lpt.values())


def test_slack_time():
    # Create a scheduler instance with dummy data
    scheduler = SurgeryScheduling(number_surgeries=50, distr="normal", number_of_ors=3)

    # Manually populate the waiting lists with test data
    scheduler.waiting_list_mixture_surgeries = {
        "surgery1": [(200, 50)],
        "surgery2": [(200, 30)],
        "surgery3": [(400, 40)]
    }
    scheduler.waiting_list_individual_surgeries = {
        "surgery1": [(200, 50), (100, 20)],
        "surgery2": [(100.5, 30), (400, 40)],
        "surgery3": [(300, 40), (205, 50)]
    }
    scheduler.weights_dictionary = {
        "surgery1": [0.6, 0.4],
        "surgery2": [0.2, 0.8],
        "surgery3": [0.5, 0.5]
    }

    # Assign surgeries to an OR (manually for testing)
    scheduler.ors = {
        1: ["surgery1", "surgery2", "surgery3"],
        2: ["surgery3"],
        3: []
    }

    # Test the slack_time function
    slack_time_value = scheduler.slack_time(1)

    # Print the result for verification
    print(f"Slack time for OR 1 with overtime probability {scheduler.overtime_prob}: {slack_time_value}")


# Run the test
test_slack_time()

# sorted_waiting_list = scheduler.sort_waiting_list()
# print(sorted_waiting_list[0])  # Print sorted list of expected durations with their keys
# print(sorted_waiting_list[1])
# print(scheduler.lpt(3))

Z = 2
# max_capacity = 300
alpha = 1

# ors_schedule_regret, total_overtime_regret = scheduler.regret_based_sampling(Z, alpha)
# print("Results for LPT Algorithm:")
# for or_num, surgeries in ors_schedule_lpt.items():
#     or_duration = sum(surgery[1] for surgery in surgeries)
#     overtime = max(0, or_duration - scheduler.max_capacity)
#     print(f"OR {or_num}: {surgeries} | Overtime: {overtime}")
# print(f"Total Overtime: {total_overtime_lpt}")
#
# print("\nResults for Regret-based Sampling Algorithm:")
# for or_num, surgeries in ors_schedule_regret.items():
#     or_duration = sum(surgery[1] for surgery in surgeries)
#     overtime = max(0, or_duration - scheduler.max_capacity)
#     print(f"OR {or_num}: {surgeries} | Overtime: {overtime}")
# print(f"Total Overtime: {total_overtime_regret}")
