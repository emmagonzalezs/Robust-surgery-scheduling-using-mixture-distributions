import pickle
import random
import numpy as np
import multiprocessing

ff_original = pickle.load(open("ff_schedule_original.pkl", 'rb'))
ff_alternate = pickle.load(open("ff_schedule_alternate.pkl", 'rb'))
ff_alternate_same_free_ors = pickle.load(open('ff_schedule_alternate.pkl', 'rb'))

lpt_original = pickle.load(open("lpt_schedule_original.pkl", 'rb'))
lpt_alternate = pickle.load(open("lpt_schedule_alternate.pkl", 'rb'))
lpt_alternate_same_free_ors = pickle.load(open('lpt_schedule_alternate.pkl', 'rb'))

regret_original = pickle.load(open("regret_schedule_original.pkl", 'rb'))
regret_alternate = pickle.load(open("regret_schedule_alternate.pkl", 'rb'))
ff_original_same_free_ors = pickle.load(open("ff_schedule_original_try_new_p.pkl", "rb"))
lpt_original_same_free_ors = pickle.load(open('lpt_schedule_original_try_new_p.pkl', "rb"))
regret_original_same_free_ors = pickle.load(open("regret_schedule_original_try_new_p.pkl", "rb"))


def generate_waiting_list_to_schedule():
    waiting_list_individual_surgeries = pickle.load(open('waiting_individual_400_surgeries_multi.pickle', 'rb'))
    waiting_list_mixture_surgeries = pickle.load(open('waiting_list_400_surgeries_multi.pickle', 'rb'))
    weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))  # {"general code": [w11,w12,w13]}
    # weights_surgeries_waiting_list = [weights_dictionary[value[0]] for key, value in
    #                                   waiting_list_mixture_surgeries.items() if
    #                                   value[0] in weights_dictionary]  # surgery in position 0
    grouped_surgeries_dict = pickle.load(open('grouped_surgeries_dict.pkl', 'rb'))
    # simulated_schedule = {(k, t): [] for t in range(1, 80 + 1) for k in range(1, 3 + 1)}
    simulated_waiting_list = {patient_id: []  for patient_id in waiting_list_mixture_surgeries}

    for patient_id, surgery in waiting_list_mixture_surgeries.items():
        probs_per_surgery = weights_dictionary[surgery[0]]
        procedures = grouped_surgeries_dict[surgery[0]]
        selected_procedure = random.choices(procedures, weights=probs_per_surgery)[0]
        selected_index = procedures.index(selected_procedure)

        individual_parameters = waiting_list_individual_surgeries[patient_id]
        mu_selected, sigma_selected = individual_parameters[0][selected_index]

        random_duration = np.random.normal(mu_selected, sigma_selected)
        info_for_or = [surgery[0], selected_procedure, random_duration]
        simulated_waiting_list[patient_id] = info_for_or

    return simulated_waiting_list


waiting_list_simulated = generate_waiting_list_to_schedule()


def schedule_simulation(generated_schedule):
    simulated_schedule = {(k, t): [] for t in range(1, 80 + 1) for k in range(1, 3 + 1)}
    # print(generated_schedule)
    for or_plan, surgeries in generated_schedule.items():
        for surgery in surgeries:
            info_for_or = [surgery[0], surgery[1], waiting_list_simulated[surgery[0]][1], waiting_list_simulated[surgery[0]][2]]
            # info_for_or = [patient_id, surgery, selected_procedure, expected_duration]
            simulated_schedule[or_plan].append(info_for_or)
    # print("simulated schedule",simulated_schedule)

    return simulated_schedule


def calculate_overtime_and_free_time(schedule):
    total_free_time = 0
    total_overtime = 0
    count_overtime = 0
    max_capacity = 450
    for or_id, or_surgeries in schedule.items():
        total_scheduled_time = sum(surgery[3] for surgery in or_surgeries)
        total_free_time += max(0, max_capacity - total_scheduled_time)
        if total_scheduled_time > max_capacity:
            total_overtime += (total_scheduled_time - max_capacity)
            count_overtime += 1
    return total_free_time, total_overtime, count_overtime


def calculate_free_days(schedule):
    total_free_days = 0
    for or_plan, surgeries in schedule.items():
        if not surgeries:
            total_free_days += 1
    return total_free_days


def run_ff():
    generated_schedule_ff_original = schedule_simulation(ff_original_same_free_ors)
    generated_schedule_ff_alternate = schedule_simulation(ff_alternate_same_free_ors)

    total_free_time_ff_original = calculate_overtime_and_free_time(generated_schedule_ff_original)
    total_free_time_ff_alternate = calculate_overtime_and_free_time(generated_schedule_ff_alternate)
    total_free_days_ff_original = calculate_free_days(generated_schedule_ff_original)
    total_free_days_ff_alternate = calculate_free_days(generated_schedule_ff_alternate)

    return total_free_time_ff_original, total_free_days_ff_original, total_free_time_ff_alternate, total_free_days_ff_alternate


def run_lpt():
    generated_schedule_lpt_original = schedule_simulation(lpt_original_same_free_ors)
    generated_schedule_lpt_alternate = schedule_simulation(lpt_alternate_same_free_ors)

    total_free_time_lpt_original = calculate_overtime_and_free_time(generated_schedule_lpt_original)
    total_free_time_lpt_alternate = calculate_overtime_and_free_time(generated_schedule_lpt_alternate)
    total_free_days_lpt_original = calculate_free_days(generated_schedule_lpt_original)
    total_free_days_lpt_alternate = calculate_free_days(generated_schedule_lpt_alternate)

    return total_free_time_lpt_original, total_free_days_lpt_original, total_free_time_lpt_alternate, total_free_days_lpt_alternate


def run_regret():
    generated_schedule_regret_original = schedule_simulation(regret_original)
    generated_schedule_regret_alternate = schedule_simulation(regret_alternate)

    total_free_time_regret_original = calculate_overtime_and_free_time(generated_schedule_regret_original)
    total_free_time_regret_alternate = calculate_overtime_and_free_time(generated_schedule_regret_alternate)
    total_free_days_regret_original = calculate_free_days(generated_schedule_regret_original)
    total_free_days_regret_alternate = calculate_free_days(generated_schedule_regret_alternate)

    return total_free_time_regret_original, total_free_days_regret_original, total_free_time_regret_alternate, total_free_days_regret_alternate


# def run_multiple_times(args):
#     func, num_runs = args
#     return [func() for _ in range(num_runs)]
#
#
# def run_simulations(functions):
#     results = [func() for func in functions]
#     return results


# def calculate_average_single(results):
#     total_free_time = [result[0][0] for result in results]
#     total_overtime = [result[0][1] for result in results]
#     count_overtime = [result[0][2] for result in results]
#     total_free_days = [result[1] for result in results]
#
#     avg_total_free_time = np.mean(total_free_time)
#     avg_total_overtime = np.mean(total_overtime)
#     avg_count_overtime = np.mean(count_overtime)
#     avg_total_free_days = np.mean(total_free_days)
#
#     return avg_total_free_time, avg_total_overtime, avg_count_overtime, avg_total_free_days
#
#
# def calculate_average(results):
#     avg_ff = calculate_average_single(results[0])
#     avg_lpt = calculate_average_single(results[1])
#     avg_regret = calculate_average_single(results[2])
#     return avg_ff, avg_lpt, avg_regret
#
#
# if __name__ == "__main__":
#     # num_runs = 100
#     functions = [run_ff, run_lpt, run_regret]
#     results = run_simulations(functions)
#     averages = calculate_average(results)
#
#     # Print the averages
#     print("FF Averages (total_free_time, total_overtime, count_overtime, total_free_days):", averages[0])
#     print("LPT Averages (total_free_time, total_overtime, count_overtime, total_free_days):", averages[1])
#     print("Regret Averages (total_free_time, total_overtime, count_overtime, total_free_days):", averages[2])
# # print("Stored ff original:", pickle.load(open("ff_schedule_original.pkl", 'rb')))
# # print("Stored regret schedule alternate:", pickle.load(open("regret_schedule_alternate.pkl", 'rb')))

# ff_run = run_ff()
# lpt_run = run_lpt()
# regret_run = run_regret()
#
# print("Simulated waiting list:", waiting_list_simulated)
#
# print("FF ORIGINAL:")
# print(f"Total free time: {ff_run[0][0]}")
# print(f"Total overtime: {ff_run[0][1]}")
# print(f"Days with overtime: {ff_run[0][2]}")
# print(f"Total free days: {ff_run[1]}")
#
# print("FF ALTERNATE")
# print(f"Total free time: {ff_run[2][0]}")
# print(f"Total overtime : {ff_run[2][1]}")
# print(f"Days with overtime : {ff_run[2][2]}")
# print(f"Total free days : {ff_run[3]}")
#
# print("LPT ORIGINAL:")
# print(f"Total free time: {lpt_run[0][0]}")
# print(f"Total overtime: {lpt_run[0][1]}")
# print(f"Days with overtime: {lpt_run[0][2]}")
# print(f"Total free days: {lpt_run[1]}")
#
# print("LPT ALTERNATE")
# print(f"Total free time: {lpt_run[2][0]}")
# print(f"Total overtime : {lpt_run[2][1]}")
# print(f"Days with overtime : {lpt_run[2][2]}")
# print(f"Total free days : {lpt_run[3]}")
#
# print("REGRET ORIGINAL:")
# print(f"Total free time: {regret_run[0][0]}")
# print(f"Total overtime: {regret_run[0][1]}")
# print(f"Days with overtime: {regret_run[0][2]}")
# print(f"Total free days: {regret_run[1]}")
#
# print("REGRET ALTERNATE")
# print(f"Total free time: {regret_run[2][0]}")
# print(f"Total overtime : {regret_run[2][1]}")
# print(f"Days with overtime : {regret_run[2][2]}")
# print(f"Total free days : {regret_run[3]}")

def run_simulations(functions, num_runs):
    results = []
    for _ in range(num_runs):
        results.append([func() for func in functions])
    return results


def calculate_average(results):
    avg_ff_original = np.mean([result[0][0] for result in results], axis=0)
    avg_ff_alternate = np.mean([result[0][2] for result in results], axis=0)
    avg_ff_original_free_days = np.mean([result[0][1] for result in results], axis=0)
    avg_ff_alternate_free_days = np.mean([result[0][3] for result in results], axis=0)

    avg_lpt_original = np.mean([result[1][0] for result in results], axis=0)
    avg_lpt_alternate = np.mean([result[1][2] for result in results], axis=0)
    avg_lpt_original_free_days = np.mean([result[1][1] for result in results], axis=0)
    avg_lpt_alternate_free_days = np.mean([result[1][3] for result in results], axis=0)

    avg_regret_original = np.mean([result[2][0] for result in results], axis=0)
    avg_regret_alternate = np.mean([result[2][2] for result in results], axis=0)
    avg_regret_original_free_days = np.mean([result[2][1] for result in results], axis=0)
    avg_regret_alternate_free_days = np.mean([result[2][3] for result in results], axis=0)

    return avg_ff_original, avg_ff_alternate, avg_ff_original_free_days, avg_ff_alternate_free_days, avg_lpt_original, avg_lpt_alternate, avg_lpt_original_free_days, avg_lpt_alternate_free_days, avg_regret_original, avg_regret_alternate, avg_regret_original_free_days, avg_regret_alternate_free_days


if __name__ == "__main__":
    num_runs = 500
    functions = [run_ff, run_lpt, run_regret]
    results = run_simulations(functions, num_runs)
    avg_ff_original, avg_ff_alternate, avg_ff_original_free_days, avg_ff_alternate_free_days, avg_lpt_original, avg_lpt_alternate, avg_lpt_original_free_days, avg_lpt_alternate_free_days, avg_regret_original, avg_regret_alternate, avg_regret_original_free_days, avg_regret_alternate_free_days = calculate_average(results)

    # functions = [lambda: run_ff()[0]]
    # results = run_simulations(functions, num_runs)
    # avg_ff_original = np.mean([result[0] for result in results], axis=0)

    # Print the averages
    # print("FF ORIGINAL:")
    # print(f"Total free time: {avg_ff_original[0] / 60} hours")
    # print(f"Total overtime: {avg_ff_original[1] / 60} hours")
    # print(f"Days with overtime: {avg_ff_original[2]}")

    # Print the averages
    print("FF ORIGINAL:")
    print(f"Total free time: {avg_ff_original[0]/60} hours")
    print(f"Total overtime: {avg_ff_original[1]/60} hours")
    print(f"Days with overtime: {avg_ff_original[2]}")
    print(f"Total free days: {avg_ff_original_free_days}")

    print("FF ALTERNATE")
    print(f"Total free time: {avg_ff_alternate[0]/60} hours")
    print(f"Total overtime : {avg_ff_alternate[1]/60}")
    print(f"Days with overtime : {avg_ff_alternate[2]}")
    print(f"Total free days : {avg_ff_alternate_free_days}")

    print("LPT ORIGINAL:")
    print(f"Total free time: {avg_lpt_original[0]/60} hours")
    print(f"Total overtime: {avg_lpt_original[1]/60} hours")
    print(f"Days with overtime: {avg_lpt_original[2]}")
    print(f"Total free days: {avg_lpt_original_free_days}")

    print("LPT ALTERNATE")
    print(f"Total free time: {avg_lpt_alternate[0]/60} hours")
    print(f"Total overtime : {avg_lpt_alternate[1]/60} hours")
    print(f"Days with overtime : {avg_lpt_alternate[2]}")
    print(f"Total free days : {avg_lpt_alternate_free_days}")

    print("REGRET ORIGINAL:")
    print(f"Total free time: {avg_regret_original[0]/60} hours")
    print(f"Total overtime: {avg_regret_original[1]/60} hours")
    print(f"Days with overtime: {avg_regret_original[2]}")
    print(f"Total free days: {avg_regret_original_free_days}")

    print("REGRET ALTERNATE")
    print(f"Total free time: {avg_regret_alternate[0]/60} hours")
    print(f"Total overtime : {avg_regret_alternate[1]/60} hours")
    print(f"Days with overtime : {avg_regret_alternate[2]}")
    print(f"Total free days : {avg_regret_alternate_free_days}")