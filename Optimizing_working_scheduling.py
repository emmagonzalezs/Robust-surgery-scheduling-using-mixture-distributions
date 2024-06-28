from WaitingList import waiting_list
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
import pickle
import random
import itertools

class SurgeryScheduling:
    def __init__(self, number_surgeries, distr, number_of_ors):
        self.number_surgeries = number_surgeries
        self.distr = distr
        self.number_of_ors = number_of_ors
        # Initialize waiting list to schedule in ORs
        self.waiting = waiting_list(number_surgeries, distr)
        # print(waiting[0])
        self.waiting_list_mixture_surgeries = self.waiting[0]  # {'1': ["general code", (mu,sigma)]}
        self.waiting_list_individual_surgeries = self.waiting[1]  # {'1': [(mu11,sigma11), (mu12, sigma12)]}

        # Weights per surgery in waiting list, for delta calculations
        self.weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))  # {"general code": [w11,w12,w13]}
        # self.weights_surgeries_waiting_list = [self.weights_dictionary[key] for key in
        #                                        self.waiting_list_mixture_surgeries if key in self.weights_dictionary]
        self.weights_surgeries_waiting_list = [self.weights_dictionary[value[0]] for key, value in
                                               self.waiting_list_mixture_surgeries.items() if
                                               value[0] in self.weights_dictionary]  # surgery in position 0
        # weights_surgeries_waiting_list = [[w11,w12,w13], [w21,w22], ...]

        # self.ors = {i: [] for i in range(1, number_of_ors + 1)}  # {"1":[], "2":[]}
        self.ors = {(k, t): [] for t in range(1, 2) for k in range(1, number_of_ors + 1)}
        # self.ors = {(OR, day):} for 1 year, append here ['surgery'
        self.max_capacity = 450  # per OR in minutes (about 8h)
        self.list_of_ors = [i for i in range(1, number_of_ors + 1)]  # [1,2,..., number_of_ors]
        self.list_of_ors = [(k, t) for t in range(1, 2) for k in range(1, number_of_ors + 1)]
        self.overtime_prob = 0.3
        self.alpha = 10

    # def sort_waiting_list(self):
    #     expected_durations_waiting = []
    #     for key, parameters in self.waiting_list_mixture_surgeries.items():
    #         for t in parameters:
    #             expected_durations_waiting.append((key, t[0]))
    #
    #     expected_durations_waiting.sort(reverse=True, key=lambda x: x[1])  # ordered [(mu,sigma)]
    #
    #     sorted_surgeries = [surgery for surgery, value in expected_durations_waiting]  # list of ordered surgeries
    #
    #     return expected_durations_waiting, sorted_surgeries

    def sort_waiting_list(self):
        ordered_waiting_list_dict = dict(sorted(self.waiting_list_mixture_surgeries.items(), key=lambda item: item[1][0], reverse=True))
        ordered_waiting_list_list = [[k, *v] for k, v in ordered_waiting_list_dict.items()]
        # only general code with mixture params here
        sorted_surgeries = list(ordered_waiting_list_dict.keys())

        return ordered_waiting_list_dict, ordered_waiting_list_list, sorted_surgeries

    def lpt(self):
        _, ordered_waiting_list, _ = self.sort_waiting_list()

        for surgery_info in ordered_waiting_list:
            empty_ors = [or_id for or_id, surgeries in self.ors.items() if not surgeries]
            if empty_ors:  # If there are empty ORs
                selected_or = empty_ors[0]
                self.ors[selected_or].append(surgery_info)  # Append the surgery to a random empty OR
            else:  # If there are no empty ORs
                min_or_used = min(self.ors, key=lambda k: sum(x[2][0] for x in self.ors[k]) + self.slack_time(self.ors[k]))
                self.ors[min_or_used].append(surgery_info)  # Append the surgery to the OR with minimum total time

        total_overtime = 0
        for or_id, or_surgeries in self.ors.items():
            # Calculate total surgery time and slack time for the OR
            total_time = sum(t[2][0] for t in or_surgeries) + self.slack_time(self.ors[or_id])
            overtime = max(total_time - self.max_capacity, 0)
            total_overtime += overtime

        return self.ors, total_overtime

    def delta_expression(self, surgeries_for_slack, delta):

        if isinstance(surgeries_for_slack[0], str):
            surgeries_for_slack = [surgeries_for_slack]
        else:
            pass

        weights_for_slack = [self.weights_dictionary[surgery[1]] for surgery in surgeries_for_slack]

        patient_id = [patient_info[0] for patient_info in surgeries_for_slack]

        mu_list = []
        sigma_list = []

        # Calculate total_mu outside the loop
        total_mu = sum(
            self.waiting_list_mixture_surgeries[patient_info[0]][1][0] for patient_info in surgeries_for_slack)

        for ids in patient_id:
            mu_sigma_pairs = self.waiting_list_individual_surgeries[ids][0]
            mu_values, sigma_values = zip(*mu_sigma_pairs)
            mu_list.append(mu_values)
            sigma_list.append(sigma_values)

        mu_list = [list(mu) for mu in mu_list]
        sigma_list = [list(sigma) for sigma in sigma_list]

        weights = np.array(weights_for_slack, dtype=object)
        mus = np.array(mu_list, dtype=object)
        sigmas = np.array(sigma_list, dtype=object)

        num_elements = [len(sublist) for sublist in weights_for_slack]

        # Generate all combinations of indices
        index_combinations = list(itertools.product(*[range(n) for n in num_elements]))

        # Extract weights, means, and sigmas based on index combinations
        selected_weights = [[weight[i] for i, weight in zip(indices, weights)] for indices in index_combinations]
        selected_mus = [[mu[i] for i, mu in zip(indices, mus)] for indices in index_combinations]
        selected_sigmas = [[sigma[i] for i, sigma in zip(indices, sigmas)] for indices in index_combinations]

        # Calculate product of weights for each combination
        weight_products = np.prod(np.array(selected_weights), axis=1)

        # Calculate sum of mus and sigmas for each combination
        mu_sums = np.sum(np.array(selected_mus), axis=1)
        sigma_sums = np.sum(np.array(selected_sigmas), axis=1)

        # Calculate the argument of the CDF
        cdf_arguments = (total_mu + delta - mu_sums) / sigma_sums

        # Compute the CDF values
        cdf_values = norm.cdf(cdf_arguments)

        # Compute the final result as the weighted sum of the CDF values
        result = np.sum(weight_products * cdf_values)

        return result

    def to_solve(self, delta, surgeries_for_slack):
        return self.delta_expression(surgeries_for_slack, delta) - (1 - self.overtime_prob)

    def slack_time(self, surgeries_for_slack):
        if not surgeries_for_slack:
            return 0

        solution_slack = fsolve(self.to_solve, x0=np.array([0]), args=surgeries_for_slack)
        return abs(solution_slack[0])

    def calculate_priority(self, current_surgery, possible_ors, schedule):  # current_surgery: [k,surgery,(mu,sigma)]
        diff_per_or = []
        omega_per_or = {}

        for ors in possible_ors:
            surgeries_with_current_surgery = []
            current_surgeries_in_or = schedule[ors]  # list of surgeries in specified OR

            if current_surgeries_in_or:
                surgeries_with_current_surgery.extend(current_surgeries_in_or)
                surgeries_with_current_surgery.append(current_surgery)
                diff_or = self.slack_time(surgeries_with_current_surgery) - self.slack_time(current_surgeries_in_or)
                diff_per_or.append(diff_or)
                omega_or = self.slack_time(current_surgery) - diff_or
            else:
                omega_or = 0

            omega_per_or[ors] = omega_or

        best_or = max(omega_per_or, key=omega_per_or.get)
        priority_current_surgery = omega_per_or[best_or]

        return current_surgery, priority_current_surgery, best_or

    def drawing_probabilities(self, priorities_list):
        all_priorities = [surgeries[1] for surgeries in priorities_list]
        all_regrets = [(surgeries[1] - min(all_priorities)) for surgeries in priorities_list]

        total = np.sum([(1 + w_i) ** self.alpha for w_i in all_regrets])
        probabilities = [(1 + regret) ** self.alpha / total for regret in all_regrets]

        return probabilities

    def regret_based_sampling(self, z, samples):
        sorted_waiting_dict, sorted_waiting_list, sorted_ops = self.sort_waiting_list()
        best_schedule = None
        best_schedule_overtime = float("inf")

        or_slack = {or_plan: 0 for or_plan in self.ors}

        for _ in range(samples):
            current_schedule = {or_plan: [] for or_plan in self.ors}
            remaining_surgeries = sorted_waiting_list[:]  # create a copy of the sorted waiting list
            while remaining_surgeries:
                # print("This many surgeries left:", len(remaining_surgeries))
                z_surgeries = remaining_surgeries[:min(z, len(remaining_surgeries))]  # take the next z surgeries
                # remaining_surgeries = remaining_surgeries[z:]  # remove the scheduled surgeries

                priorities_z_surgeries = []  # initialize priorities list for z surgeries
                surgeries_to_draw = []
                for surgery in z_surgeries:  # surgery = ["id", surgery, (mu,sigma)]
                    possible_ors = []
                    for or_plan, value in current_schedule.items():
                        surgeries_in_or = current_schedule[or_plan]
                        # if the OR is not empty:
                        if surgeries_in_or:
                            surgeries_or_with_surgery = surgeries_in_or + [surgery]  # find slack for all together
                            # slack = self.slack_time(surgeries_or_with_surgery)
                            slack = self.slack_time(surgeries_or_with_surgery)
                            if sum(t[2][0] for t in current_schedule[or_plan]) + surgery[2][0] + \
                                    slack < self.max_capacity:
                                possible_ors.append(or_plan)
                                # print("calculated_slack:", slack)
                        # If empty, no need to check, it is a possible OR
                        else:
                            possible_ors.append(or_plan)

                    if possible_ors:
                        priority_surgery_i = self.calculate_priority(surgery, possible_ors, current_schedule)
                        # priority_surgery_i = ["id",surgery,(mu,sigma)], priority, best_or
                        priorities_z_surgeries.append(priority_surgery_i)
                        surgeries_to_draw.append(priority_surgery_i[0])  # whole surgery info, ["id",surgery,(mu,sigma)]

                    else:
                        # min_or_used = min(current_schedule.keys(), key=lambda k: sum(t[2][0] for t in
                        #                   current_schedule[k]) + surgery[2][0] +
                        #                   self.slack_time(current_schedule[k] + [surgery]) - self.max_capacity)
                        min_or_used = min(current_schedule.keys(), key=lambda k: sum(t[2][0] for t in
                                          current_schedule[k]) + surgery[2][0] + self.slack_time(current_schedule[k]) -
                                          self.max_capacity)
                        current_schedule[min_or_used].append(surgery)
                        # print(f"surgery {surgery[0]} scheduled in {min_or_used}")
                        remaining_surgeries.remove(surgery)  # remove the surgery from the remaining list
                        # print(f'Surgery {surgery} did not fit without overtime, so scheduled in {min_or_used}')

                if priorities_z_surgeries:
                    probabilities = self.drawing_probabilities(priorities_z_surgeries)
                    drawn_surgery = random.choices(surgeries_to_draw, probabilities, k=1)[0]
                    # drawn_surgery: list of info of the surgery drawn
                    for i, (surgery, priority, best_or) in enumerate(priorities_z_surgeries):
                        if surgery == drawn_surgery:  # patient info (list)
                            current_schedule[best_or].append(surgery)
                            # print(f"surgery {surgery[0]} scheduled in {best_or}")
                            # print("drawn surgery", drawn_surgery_info)
                            # print(remaining_surgeries)
                            remaining_surgeries.remove(surgery)  # remove the scheduled surgery

                            break

            total_overtime = 0

            for or_id, or_surgeries in current_schedule.items():
                overtime = max(sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries) - self.max_capacity,
                               0)
                total_overtime += overtime
            #     print(f'OR {or_id}: overtime = {overtime}')
            #     print(f'slack for OR {or_id}: = {self.slack_time(or_surgeries)}')
            # print("total overtime", total_overtime)

            # Update best schedule if overtime is less
            if total_overtime < best_schedule_overtime:
                best_schedule = current_schedule
                best_schedule_overtime = total_overtime

        return best_schedule, best_schedule_overtime


surgery_scheduling = SurgeryScheduling(number_surgeries=10, distr="normal", number_of_ors=4)
start_time1 = time.time()
best_schedule_output, best_schedule_overtime_output = surgery_scheduling.regret_based_sampling(z=2, samples=5)
end_time1 = time.time()
regret_based_sampling_time = end_time1 - start_time1

start_time2 = time.time()
lpt_schedule, overtime_lpt = surgery_scheduling.lpt()
end_time2 = time.time()
lpt_time = end_time2 - start_time2

print("Best schedule using regret based:", best_schedule_output)
print(f"Total overtime using regret based: {best_schedule_overtime_output}, with z=2, samples=5, ORs=4")
print("Total running time for regret based:", regret_based_sampling_time)

print("Best schedule using lpt:", lpt_schedule)
print("Total overtime using lpt:", overtime_lpt)
print("Total running time for lpt:", lpt_time)

