import numpy as np
from scipy.stats import norm

def phi(x):
    return norm.cdf(x)

weights = np.array([[w11, w12, w13], [w21, w22, w23], [w31, w32, w33],...])
mus = np.array([[mu11, mu12, mu13], [mu21, mu22, mu23], [mu31, mu32, mu33],...])
sigmas = np.array([[s11, s12, s13], [s21, s22, s23], [s31, s32, s33],...])

result = np.sum(weights[:, None, :] * weights[None, :, :] *
                 phi((mus[:, None, :] + mus[None, :, :]) / (sigmas[:, None, :] + sigmas[None, :, :])))


########### CURRENT delta_expression function:

# def delta_expression(self, surgeries_for_slack, delta):
#     # print("surgeries for slack in delta expr", surgeries_for_slack)
#     # Step 1: Initialize dictionaries
#     # mixture_params_for_slack = {surgery: self.waiting_list_mixture_surgeries[surgery[0]] for surgery in
#     #                             surgeries_for_slack}  # {"general code": [mu,sigma]}
#     # individual_params_for_slack = {surgery: self.waiting_list_individual_surgeries[surgery[0]] for surgery in
#     #                                surgeries_for_slack}  # {"general code": [(mu11,sigma11), (mu12, sigma12)]}
#     # weights_for_slack = {surgery: self.weights_dictionary[surgery] for surgery in surgeries_for_slack}
#     # print("surgeries for slack", surgeries_for_slack)
#     # print("waiting list mixture surgeries", self.waiting_list_mixture_surgeries)
#     mixture_params_for_slack = []
#     individual_params_for_slack = []
#     weights_for_slack = []
#
#     if len(surgeries_for_slack) == 3 and type(surgeries_for_slack[0]) == str:
#         surgery_k = str(surgeries_for_slack[0])
#         # print(surgery_k)
#         # print(self.waiting_list_mixture_surgeries)
#         mixture_params_for_slack.append(self.waiting_list_mixture_surgeries[surgery_k])
#         individual_params_for_slack.append(self.waiting_list_individual_surgeries[surgery_k])
#         # [[(mu11,sigma11), (mu12,sigma12),...],...]
#         surgery_ops = surgeries_for_slack[1]
#         # print(surgery_ops)
#         weights_for_slack.append(self.weights_dictionary[surgery_ops])
#     else:
#         for surg in surgeries_for_slack:
#             # print("surgeries for slack", surgeries_for_slack)
#             # print("surg", surg)
#             surgery_k = surg[0]
#             mixture_params_for_slack.append(self.waiting_list_mixture_surgeries[surgery_k])
#             individual_params_for_slack.append(self.waiting_list_individual_surgeries[surgery_k])
#             # [[(mu11,sigma11), (mu12,sigma12),...],...]
#             surgery_ops = surg[1]
#             # print(surgery_ops)
#             weights_for_slack.append(self.weights_dictionary[surgery_ops])
#
#             if len(individual_params_for_slack) == 1:
#                 individual_params_for_slack[0] = [(mu, sigma) for mu, sigma in individual_params_for_slack[0][0]]
#
#     # mixture_params_for_slack = [self.waiting_list_mixture_surgeries[k] for k, _, _ in
#     #                             surgeries_for_slack]  # [["general code, (mu,sigma),...]
#
#     # Step 2: Compute total mu per OR
#     total_mu_per_or = sum(surgery[1][0] for surgery in mixture_params_for_slack)
#     # print("total_mu", total_mu_per_or)
#
#     # Step 3: Generate weight combinations and their products so w11*w21, w11*w22 etc.
#     num_procedures_per_surgery = [len(weights) for weights in weights_for_slack]
#     if len(num_procedures_per_surgery) > 1:
#         # print("weights for slack", weights_for_slack)
#         # print("num procedures", num_procedures_per_surgery)
#         products_weights = np.array(
#             [np.prod([weights_for_slack[j][index] for j, index in enumerate(procedure)])
#              for procedure in product(*[range(n) for n in num_procedures_per_surgery])])
#         # Step 4: Prepare individual parameters as arrays
#         individual_params_list = [np.array(params) for params in individual_params_for_slack]
#         # print("individual_params_list", individual_params_list)
#         # Step 5: Generate combinations of indices for mu and s values
#         # index_combinations = list(product(*[range(len(params)) for params in individual_params_list]))
#         index_combinations = list(product(*[range(n) for n in num_procedures_per_surgery]))
#         # Step 6: Compute sum of mu and s for each combination
#         # print("individual params list", individual_params_list)
#         # sum_mu_array = np.array(
#         #     [sum(individual_params_list[i][index][0] for i, index in enumerate(indices)) for indices in
#         #      index_combinations])
#         # sum_s_array = np.array(
#         #     [sum(individual_params_list[i][index][1] for i, index in enumerate(indices)) for indices in
#         #      index_combinations])
#         # print("surgeries for slack", surgeries_for_slack)
#         sum_mu_array = np.array([sum(individual_params_list[i][0, index, 0] for i, index in enumerate(indices))
#             for indices in index_combinations])
#         # print("sum mu array", sum_mu_array)
#         # Calculate the sum of the second elements in each tuple across all combinations of indices
#         sum_s_array = np.array([sum(individual_params_list[i][0, index, 1] for i, index in enumerate(indices))
#                                 for indices in index_combinations])
#         # print("sum s array", sum_s_array)
#         # Step 7: Compute phi values for each combination
#         # phi_array = norm.cdf((total_mu_per_or + delta - sum_mu_array) / sum_s_array)
#         phi_array = np.array([norm.cdf((total_mu_per_or + delta - mu) / s)
#                               for mu, s in zip(sum_mu_array, sum_s_array)])
#         # print("phi", phi_array)
#         # print("products weights", products_weights)
#
#         # Step 8: Calculate the final delta expression using dot product
#         delta_expression_value = np.dot(products_weights, phi_array)
#         # print("delta", delta_expression_value)
#
#     else:  # if only one surgery, do not find product of all
#         # print(num_procedures_per_surgery)
#         # print(individual_params_for_slack)
#         # print(individual_params_for_slack)
#         mu_list = [surgery[0] for surgery in individual_params_for_slack[0][0][0]]
#
#         # print("mu list", mu_list)
#         sigma_list = [surgery[1] for surgery in individual_params_for_slack[0][0]]
#         # print("sigma list", sigma_list)
#         phi = norm.cdf((total_mu_per_or + delta - np.array(mu_list))/ np.array(sigma_list))
#         # print(phi)
#         # print(weights_for_slack)
#         delta_expression_value = np.dot(weights_for_slack, phi)
#
#     return delta_expression_value

def delta_expression(self, surgeries_for_slack, delta):
    # Precompute weights and mixture parameters
    # print(surgeries_for_slack)
    # print(self.weights_dictionary)
    if type(surgeries_for_slack[0]) == str:
        # print("if part", surgeries_for_slack[0])
        weights_for_slack = [self.weights_dictionary[surgeries_for_slack[1]]]
        mixture_params_for_slack = [self.waiting_list_mixture_surgeries[surgeries_for_slack[0]]]
        individual_params_for_slack = [list(self.waiting_list_individual_surgeries[surgeries_for_slack[0]])]
    else:
        # print("else", surgeries_for_slack[0])
        weights_for_slack = [self.weights_dictionary[surgeries_for_slack[0][1]]]
        mixture_params_for_slack = [self.waiting_list_mixture_surgeries[surgeries_for_slack[0][0]]]
        individual_params_for_slack = [list(self.waiting_list_individual_surgeries[surgeries_for_slack[0][0]])]

    # Compute total mu per OR
    total_mu_per_or = sum(surgery[1][0] for surgery in mixture_params_for_slack)

    # Generate weight combinations and their products
    num_procedures_per_surgery = [len(weights) for weights in weights_for_slack]
    if len(num_procedures_per_surgery) > 1:
        products_weights = np.array(
            [np.prod([weights_for_slack[j][index] for j, index in enumerate(procedure)])
             for procedure in product(*[range(n) for n in num_procedures_per_surgery])])

        # Prepare individual parameters as arrays
        individual_params_list = [np.array(params) for params in individual_params_for_slack]

        # Generate combinations of indices for mu and s values
        index_combinations = list(product(*[range(n) for n in num_procedures_per_surgery]))

        # Compute sum of mu and s for each combination
        sum_mu_array = np.array([np.sum(individual_params_list[i][:, index, 0] for i, index in enumerate(indices))
                                 for indices in index_combinations])
        sum_s_array = np.array([np.sum(individual_params_list[i][:, index, 1] for i, index in enumerate(indices))
                                for indices in index_combinations])

        # Compute
        # Step 7: Compute phi values for each combination
        # phi_array = norm.cdf((total_mu_per_or + delta - sum_mu_array) / sum_s_array)
        phi_array = np.array([norm.cdf((total_mu_per_or + delta - mu) / s)
                              for mu, s in zip(sum_mu_array, sum_s_array)])
        # print("phi", phi_array)
        # print("products weights", products_weights)

        # Step 8: Calculate the final delta expression using dot product
        delta_expression_value = np.dot(products_weights, phi_array)
        # print("delta", delta_expression_value)

    else:  # if only one surgery, do not find product of all
        # print(num_procedures_per_surgery)
        # print(individual_params_for_slack)
        # print(individual_params_for_slack)
        mu_list = [surgery[0] for surgery in individual_params_for_slack[0][0]]

        # print("mu list", mu_list)
        sigma_list = [surgery[1] for surgery in individual_params_for_slack[0][0]]
        # print("sigma list", sigma_list)
        phi = norm.cdf((total_mu_per_or + delta - np.array(mu_list)) / np.array(sigma_list))
        # print(phi)
        # print(weights_for_slack)
        delta_expression_value = np.dot(weights_for_slack, phi)

    return delta_expression_value