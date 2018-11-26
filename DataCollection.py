import os
from os.path import isfile, join
from os import listdir
import scipy.stats as ss
import numpy
import pickle
import pandas as pd
import math
import hv as HyperVolume


def collect_data(directory):
    """
    Go to each benchmark directory, for each combination in the algorithm/archive strategy/dimension directory.
    Calculate the performance measures for each run.
    Find the standard deviation and average performance measure for each run
    :return:
    """
    # we need to read in the files from the directory
    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    for algotihm_folder in algorithm_folders:
        algotihm_folder_directory = directory + "/" + algotihm_folder
        archive_strategy_folders = [name for name in os.listdir(algotihm_folder_directory) if os.path.isdir(os.path.join(algotihm_folder_directory, name))]
        for archive_strategy_folder in archive_strategy_folders:
            archive_strategy_folder_directory = algotihm_folder_directory + "/" + archive_strategy_folder
            dimensions_folders = [name for name in os.listdir(archive_strategy_folder_directory) if os.path.isdir(os.path.join(archive_strategy_folder_directory, name))]
            dimensions_list = list()
            for dimension_folder in dimensions_folders:
                dimension_folder_directory = archive_strategy_folder_directory + "/" + dimension_folder
                combination_folders = combination_sort([name for name in os.listdir(dimension_folder_directory) if os.path.isdir(os.path.join(dimension_folder_directory, name))])
                combinations_list = list()
                for combination_folder in combination_folders:
                    combination_folder_directory = dimension_folder_directory + "/" + combination_folder
                    benchmark_objectives_folders = benchmark_sort([name for name in os.listdir(combination_folder_directory) if os.path.isdir(os.path.join(combination_folder_directory, name))])
                    index = 0
                    benchmark_objective_list = list()
                    benchmarks_list = list()
                    for benchmark_objectives_folder in benchmark_objectives_folders:
                        benchmark_objective_folder_directory = combination_folder_directory + "/" + benchmark_objectives_folder
                        run_list = collect_run_results(benchmark_objective_folder_directory)
                        benchmark_objective_list.append(run_list)
                        # if this is an obj_2
                        if index % 2 != 0:
                            benchmarks_list.append(benchmark_objective_list)
                            benchmark_objective_list = list()
                        index += 1
                    combinations_list.append(benchmarks_list)
                dimensions_list.append(combinations_list)
            # archive_strategy_list.append(dimensions_list)
            pickle_results(dimensions_list, archive_strategy_folder_directory+"_result_matrix.pkl")
        # algorithm_list.append(archive_strategy_list)
        # pickle_results(archive_strategy_list, algotihm_folder)
    # pickle_results(algorithm_list)
    return


def collect_data_specific(directory, algorithm, strategies, dimension_name, dimension, param_comb_name, param_comb, benchmarks):
    """
    Go to each benchmark directory, for each combination in the algorithm/archive strategy/dimension directory.
    Calculate the performance measures for each run.
    Find the standard deviation and average performance measure for each run
    :return:
    """
    # load the result matrix
    old_results = load_results(directory + "/" + algorithm[0] + "/" + strategies[0])

    for dimension in dimension:
        for parameter_combination in param_comb:
            for benchmark in range(len(benchmarks)):
                benchmark_objective_folder_directory = directory + "/" + algorithm[0] + "/" + strategies[0][:-18] + "/" + dimension_name[0] + "/" + param_comb_name[0] + "/" + "bench_" + str(benchmarks[benchmark]) + "_obj_1"
                benchmark_objective_folder_directory_2 = directory + "/" + algorithm[0] + "/" + strategies[0][:-18] + "/" + dimension_name[0] + "/" + param_comb_name[0] + "/" + "bench_" + str(benchmarks[benchmark]) + "_obj_2"
                # read in the new data
                run_list_1 = collect_run_results(benchmark_objective_folder_directory)
                run_list_2 = collect_run_results(benchmark_objective_folder_directory_2)
                # update the old result at this position
                old_results[dimension][parameter_combination][benchmarks[benchmark]] = [run_list_1, run_list_2]

            pickle_results(old_results, "../Dynamic POF"+"/"+algorithm[0]+"/"+strategies[0])
    # pickle_results(algorithm_list)
    return


def collect_run_results(parent_directory):
    """
    Read in each archive file under each run folder under the parent directory
    :param parent_directory:
    :return:
    """
    run_list = list()
    run_folders = run_sort([name for name in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, name))])
    for run_folder in run_folders:
        archive_list = list()
        run_folder_directory = parent_directory + "/" + run_folder
        for count, archive_file in enumerate(sorted([file for file in listdir(run_folder_directory) if isfile(join(run_folder_directory, file))]), start=0):
            archive_file_directory = run_folder_directory + "/" + "archive_" + str(count)
            archive = open(archive_file_directory, "r").read().split('\n')[:-1]
            archive_list.append(archive)
        run_list.append(archive_list)
    return run_list


def find_reference_vectors(directory):
    """
    For each benchmark function: search through all runs of all tests and find the worst reference point
    :param result_matrix:
    :return:
    """
    # setup the reference points
    x_reference_point_list = list()
    y_reference_point_list = list()

    # for each benchmark function, have a max and min reference
    for loop in range(16):
        x_reference_point_list.append(0)
        y_reference_point_list.append(0)

    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    for algotihm_folder in algorithm_folders:
        algotihm_folder_directory = directory + "/" + algotihm_folder
        archive_strategies = [file for file in listdir(algotihm_folder_directory) if isfile(join(algotihm_folder_directory, file))]
        for archive_strategy in archive_strategies:
            result_matrix = load_results(algotihm_folder_directory+"/"+archive_strategy)
            for dimension in range(len(result_matrix)):
                for param_combination in range(len(result_matrix[dimension])):
                    for benchmark in range(len(result_matrix[dimension][param_combination])):
                        for run in range(len(result_matrix[dimension][param_combination][benchmark][0])):
                            for archive in range(len(result_matrix[dimension][param_combination][benchmark][0][run])):
                                # check if the max x point is greater than the current one for the benchmark
                                if result_matrix[dimension][param_combination][benchmark][0][run][archive]:
                                    archive_max_x = float(max(result_matrix[dimension][param_combination][benchmark][0][run][archive]))
                                else:
                                    archive_max_x = 0
                                if result_matrix[dimension][param_combination][benchmark][1][run][archive]:
                                    archive_max_y = float(max(result_matrix[dimension][param_combination][benchmark][1][run][archive]))
                                else:
                                    archive_max_y = 0
                                if archive_max_x > x_reference_point_list[benchmark]:
                                    x_reference_point_list[benchmark] = archive_max_x
                                if archive_max_y > y_reference_point_list[benchmark]:
                                    y_reference_point_list[benchmark] = archive_max_y
                                # check if the max y point is greater than the current one for the benchmark
    print("x reference points")
    print(x_reference_point_list)
    print("y reference points")
    print(y_reference_point_list)
    # pickle x_max
    pickle_results(x_reference_point_list, "x_max_reference")
    # pickle y_max
    pickle_results(y_reference_point_list, "y_max_reference")
    return


def save_true_pof(directory):
    # generate a list of x values
    default_x_list = list()
    for loop in range(1000):
        default_x_list.append(loop/1000)

    param_combination_list = list()
    param_combination_folders = combination_sort([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])
    for param_combination_folder in param_combination_folders:
        param_combination_folder_directory = directory + "/" + param_combination_folder
        benchmark_list = list()
        benchmark_folders = benchmark_sort_2([name for name in os.listdir(param_combination_folder_directory) if os.path.isdir(os.path.join(param_combination_folder_directory, name))])
        for benchmark_folder in benchmark_folders:
            benchmark_folder_directory = param_combination_folder_directory + "/" + benchmark_folder
            archives_names = [file for file in listdir(benchmark_folder_directory) if isfile(join(benchmark_folder_directory, file))]
            x_count = 0
            archive_list = list()
            for archive_name in archives_names:
                archive_file_directory = benchmark_folder_directory + "/pof_" + str(x_count)
                archive_x_file_directory = benchmark_folder_directory + "/pof_x_" + str(x_count)
                # if the archive name does not have an x in it
                if 'x' not in archive_name:
                    # search for the arhive name with the x, if it does not exist
                    if not os.path.isfile(archive_x_file_directory):
                        # generate use the generated list
                        archive_x = default_x_list
                    else:
                        # use the x_pof
                        archive_x = numpy.asarray(open(archive_x_file_directory, "r").read().split('\n')[:-1])
                    archive = open(archive_file_directory, "r").read().split('\n')[:-1]
                else:
                    break
                x_count += 1
                archive_list.append([archive, archive_x])
            benchmark_list.append(archive_list)
        param_combination_list.append(benchmark_list)
    pickle_results(param_combination_list, "true_pof")
    return


def read_ns(directory):
    # we have to check if the archive solutions are non-dominated for the first archive management strategy
    # calculate the average over all archives
    # calculate the average over all runs

    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    algorithm_values = list()
    for algotihm_folder in algorithm_folders:
        algotihm_folder_directory = directory + "/" + algotihm_folder
        archive_strategies = [file for file in listdir(algotihm_folder_directory) if isfile(join(algotihm_folder_directory, file))]
        archive_strategy_index = -1
        archive_strategy_average_values = list()
        for archive_strategy in archive_strategies:
            archive_strategy_index += 1
            result_matrix = load_results(algotihm_folder_directory + "/" + archive_strategy)
            dimension_average_values = list()
            for dimension in range(len(result_matrix)):
                combination_average_values = list()
                for param_combination in range(len(result_matrix[dimension])):
                    benchmark_average_values = list()
                    for benchmark in range(len(result_matrix[dimension][param_combination])):
                        run_average_list = list()
                        for run in range(len(result_matrix[dimension][param_combination][benchmark][0])):
                            archive_average_list = list()
                            for archive in range(len(result_matrix[dimension][param_combination][benchmark][0][run])):
                                archive_x = result_matrix[dimension][param_combination][benchmark][0][run][archive]
                                archive_y = result_matrix[dimension][param_combination][benchmark][1][run][archive]
                                # if we are using archive strategy 1
                                if archive_strategy_index == 0:
                                    # calculate the number of non-dominated solutions in this archive
                                    # start with the first value
                                    archive_size = max(len(archive_x), len(archive_y))
                                    ns_count = 0
                                    for archive_value_index in range(archive_size):
                                        if not archive_x:
                                            break
                                        elif not archive_y:
                                            break
                                        # we have the x and y value
                                        # check this value combination against every other value combination in the archive
                                        archive_x_value = archive_x[archive_value_index]
                                        archive_y_value = archive_y[archive_value_index]
                                        archive_value_index_2 = (archive_value_index+1) % archive_size
                                        non_dominated = True
                                        while archive_value_index_2 != archive_value_index:
                                            if archive_x[archive_value_index_2] < archive_x_value and archive_y[archive_value_index_2] < archive_y_value:
                                                non_dominated = False
                                                break
                                            archive_value_index_2 = (archive_value_index_2+1) % archive_size
                                        if non_dominated:
                                            ns_count += 1
                                    archive_average_list.append(ns_count)
                                else:
                                    # get the total number of solutions in the archive
                                    archive_average_list.append(len(archive_x))
                            # calculate the average ns for all archives of this run
                            run_average = sum(archive_average_list)/len(archive_average_list)
                            # add to the list of all runs for this benchmark
                            run_average_list.append(run_average)
                        benchmark_average = sum(run_average_list) / len(run_average_list)
                        benchmark_average_values.append(benchmark_average)
                    combination_average_values.append(benchmark_average_values)
                dimension_average_values.append(combination_average_values)
            archive_strategy_average_values.append(dimension_average_values)
        algorithm_values.append(archive_strategy_average_values)
    # result_matrix.append(archive_strategy_average_values)
    pickle_results(algorithm_values, "ns_raw")
    return


def stats_ns():
    result_matrix = load_results("ns_raw")
    # we need to tally wins and losses for each
    ns_alg_1_matrix = list()
    ns_alg_2_matrix = list()
    ns_alg_1_matrix.append(numpy.asarray(["D", "PM", "Results", "A1", "A2", "A3", "A4"]))
    ns_alg_2_matrix.append(numpy.asarray(["D", "PM", "Results", "A1", "A2", "A3", "A4"]))
    for dimension in range(len(result_matrix[0][0])):
        dimension_wins = [0, 0, 0, 0, 0, 0, 0, 0]
        dimension_losses = [0, 0, 0, 0, 0, 0, 0, 0]
        # calculate the wins/losses/difference/rank for each strategy
        # for each parameter combination
        for param_combination in range(len(result_matrix[0][0][dimension])):
            # for each benchmark
            for benchmark in range(len(result_matrix[0][0][dimension][param_combination])):
                # for both algorithms
                alg_index = 0
                for algorithm_index in range(len(result_matrix)):
                    # identify which archive management strategy wins for this benchmark
                    # get dimension name
                    if dimension == 0:
                        dimension_name = "Large"
                    elif dimension == 1:
                        dimension_name = "Low"
                    elif dimension == 2:
                        dimension_name = "Medium"
                    largest_ns = -1
                    largest_ns_index = -1
                    # for each archive management strategy
                    for archive_strategy in range(len(result_matrix[algorithm_index])):
                        # find the largest ns for this parameter/benchmark combination
                        current_benchmark = result_matrix[algorithm_index][archive_strategy][dimension][param_combination][benchmark]
                        if current_benchmark > largest_ns:
                            largest_ns = current_benchmark
                            largest_ns_index = archive_strategy
                    # allocate the winning strategy a win and the others a loss
                    for archive_strategy in range(len(result_matrix[algorithm_index])):
                        if archive_strategy == largest_ns_index:
                            # allocate a win
                            dimension_wins[archive_strategy+alg_index] += 1
                        else:
                            # allocate a loss
                            dimension_losses[archive_strategy+alg_index] += 1
                    alg_index += 4

        # create the difference
        dimension_difference = list()
        for value_index in range(len(dimension_wins)):
            dimension_difference.append(dimension_wins[value_index]-dimension_losses[value_index])
        # create the ranked list
        dimension_ranks = ss.rankdata(numpy.negative(dimension_wins), method='min')

        # save to algorithm 1 matrix
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "NS", "Wins", dimension_wins[0], dimension_wins[2], dimension_wins[1], dimension_wins[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "NS", "Losses", dimension_losses[0], dimension_losses[2], dimension_losses[1], dimension_losses[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "NS", "Diff", dimension_difference[0], dimension_difference[2], dimension_difference[1], dimension_difference[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "NS", "Rank", dimension_ranks[0], dimension_ranks[2], dimension_ranks[1], dimension_ranks[3]]))

        # save to algorithm 2 matrix
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "NS", "Wins", dimension_wins[4], dimension_wins[6], dimension_wins[5], dimension_wins[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "NS", "Losses", dimension_losses[4], dimension_losses[6], dimension_losses[5], dimension_losses[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "NS", "Diff", dimension_difference[4], dimension_difference[6], dimension_difference[5], dimension_difference[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "NS", "Rank", dimension_ranks[4], dimension_ranks[6], dimension_ranks[5], dimension_ranks[7]]))

    writer = pd.ExcelWriter('Results/ns_results.xlsx')
    df1 = pd.DataFrame(numpy.asarray(ns_alg_1_matrix))
    df2 = pd.DataFrame(numpy.asarray(ns_alg_2_matrix))
    df1.to_excel(writer, 'Sheet1')
    df2.to_excel(writer, 'Sheet2')
    writer.save()
    return


def read_acc(directory):
    print("reading acc")
    # load the reference vectors
    ref_x = load_results("x_max_reference")
    ref_y = load_results("y_max_reference")

    # load the TRUE POF's for each benchmark function
    true_pof_hv = load_results("true_pof_hv")

    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    algorithm_values = list()
    algorithm_combination = list()
    for algotihm_folder in range(len(algorithm_folders)):
        print("algotihm_folder: "+str(algorithm_folders[algotihm_folder]))
        algotihm_folder_directory = directory + "/" + algorithm_folders[algotihm_folder]
        archive_strategies = [file for file in listdir(algotihm_folder_directory) if isfile(join(algotihm_folder_directory, file))]
        archive_strategy_index = -1
        archive_strategy_average_values = list()
        archive_strategy_average_combination = list()
        for archive_strategy in range(len(archive_strategies)):
            print("archive_strategy: "+str(archive_strategies[archive_strategy]))
            archive_strategy_index += 1
            result_matrix = load_results(algotihm_folder_directory + "/" + archive_strategies[archive_strategy])
            dimension_average_values = list()
            dimension_average_combination = list()
            for dimension in range(len(result_matrix)):
                print("dimension: "+str(dimension))
                combination_average_values = list()
                combination_average_combination = list()
                for param_combination in range(1, len(result_matrix[dimension])):
                    print("param_combination: "+str(param_combination))
                    benchmark_average_values = list()
                    benchmark_average_combination = list()
                    for benchmark in range(len(result_matrix[dimension][param_combination])):
                        # print("benchmark: "+str(benchmark))
                        run_average_list = list()
                        run_list_combinations = list()
                        for run in range(len(result_matrix[dimension][param_combination][benchmark][0])):
                            archive_average_list = list()
                            for archive in range(len(result_matrix[dimension][param_combination][benchmark][0][run])):
                                # we are at an archive
                                archive_x = list(map(float, result_matrix[dimension][param_combination][benchmark][0][run][archive]))
                                archive_y = list(map(float, result_matrix[dimension][param_combination][benchmark][1][run][archive]))

                                if not archive_y or not archive_x:
                                    continue
                                if max(archive_x) == 0 or max(archive_y) == 0:
                                    continue

                                # calculate the HVD for each pof
                                # get the reference point for this benchmark function
                                referencePoint = [ref_x[benchmark], ref_y[benchmark]]

                                hv = HyperVolume.HyperVolume(referencePoint)
                                # calculate the HV of the current archive
                                # convert the archive arrays into a front
                                front = list()
                                check = True

                                for point_x, point_y in zip(archive_x, archive_y):
                                    # check if the point_y is not already in the front
                                    for elem in front:
                                        if round(point_x, 10) == elem[0] or round(point_y, 10) == elem[1]:
                                            check = False
                                    if check:
                                        front.append([round(point_x, 10), round(point_y, 10)])
                                    check = True

                                pof_volume = hv.compute(front)

                                # calculate the HV of the true POF for the param_combination/benchmark function
                                # convert the archive arrays into a front
                                front = list()
                                check = True

                                true_pof_volume = true_pof_hv[param_combination][benchmark][archive % len(true_pof_hv[param_combination][benchmark])]
                                hvd = math.fabs(true_pof_volume - pof_volume)
                                archive_average_list.append(hvd)

                            # calculate the average ns for all archives of this run
                            if archive_average_list:
                                run_average = sum(archive_average_list) / len(archive_average_list)
                                # add to the list of all runs for this benchmark
                                run_average_list.append(run_average)
                                run_list_combinations.append(archive_average_list)
                        if run_average_list:
                            benchmark_average = sum(run_average_list) / len(run_average_list)
                        else:
                            benchmark_average = 0
                            print("Empty list at: benchmark: " + str(benchmark))
                        benchmark_average_values.append(benchmark_average)
                        benchmark_average_combination.append(run_list_combinations)
                    combination_average_values.append(benchmark_average_values)
                    combination_average_combination.append(benchmark_average_combination)
                dimension_average_values.append(combination_average_values)
                dimension_average_combination.append(combination_average_combination)
            archive_strategy_average_values.append(dimension_average_values)
            archive_strategy_average_combination.append(dimension_average_combination)
        algorithm_values.append(archive_strategy_average_values)
        algorithm_combination.append(archive_strategy_average_combination)
    pickle_results(algorithm_values, "acc_raw")
    pickle_results(algorithm_combination, "acc_combinations_raw")
    return


def stats_acc():
    result_matrix = load_results("acc_raw")
    # we need to tally wins and losses for each
    ns_alg_1_matrix = list()
    ns_alg_2_matrix = list()
    ns_alg_1_matrix.append(numpy.asarray(["D", "PM", "Results", "A1", "A2", "A3", "A4"]))
    ns_alg_2_matrix.append(numpy.asarray(["D", "PM", "Results", "A1", "A2", "A3", "A4"]))
    for dimension in range(len(result_matrix[0][0])):
        dimension_wins = [0, 0, 0, 0, 0, 0, 0, 0]
        dimension_losses = [0, 0, 0, 0, 0, 0, 0, 0]
        # calculate the wins/losses/difference/rank for each strategy
        # for each parameter combination
        for param_combination in range(len(result_matrix[0][0][dimension])):
            # for each benchmark
            for benchmark in range(len(result_matrix[0][0][dimension][param_combination])):
                # for both algorithms
                alg_index = 0
                for algorithm_index in range(len(result_matrix)):
                    # identify which archive management strategy wins for this benchmark
                    # get dimension name
                    if dimension == 0:
                        dimension_name = "Large"
                    elif dimension == 1:
                        dimension_name = "Low"
                    elif dimension == 2:
                        dimension_name = "Medium"
                    largest_ns = -1
                    largest_ns_index = -1
                    # for each archive management strategy
                    for archive_strategy in range(len(result_matrix[algorithm_index])):
                        # find the largest ns for this parameter/benchmark combination
                        current_benchmark = result_matrix[algorithm_index][archive_strategy][dimension][param_combination][benchmark]
                        if current_benchmark > largest_ns:
                            largest_ns = current_benchmark
                            largest_ns_index = archive_strategy
                    # allocate the winning strategy a win and the others a loss
                    for archive_strategy in range(len(result_matrix[algorithm_index])):
                        if archive_strategy == largest_ns_index:
                            # allocate a win
                            dimension_wins[archive_strategy+alg_index] += 1
                        else:
                            # allocate a loss
                            dimension_losses[archive_strategy+alg_index] += 1
                    alg_index += 4

        # create the difference
        dimension_difference = list()
        for value_index in range(len(dimension_wins)):
            dimension_difference.append(dimension_wins[value_index]-dimension_losses[value_index])
        # create the ranked list
        dimension_ranks = ss.rankdata(numpy.negative(dimension_wins), method='min')

        # save to algorithm 1 matrix
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "acc", "Wins", dimension_wins[0], dimension_wins[2], dimension_wins[1], dimension_wins[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "acc", "Losses", dimension_losses[0], dimension_losses[2], dimension_losses[1], dimension_losses[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "acc", "Diff", dimension_difference[0], dimension_difference[2], dimension_difference[1], dimension_difference[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "acc", "Rank", dimension_ranks[0], dimension_ranks[2], dimension_ranks[1], dimension_ranks[3]]))

        # save to algorithm 2 matrix
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "acc", "Wins", dimension_wins[4], dimension_wins[6], dimension_wins[5], dimension_wins[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "acc", "Losses", dimension_losses[4], dimension_losses[6], dimension_losses[5], dimension_losses[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "acc", "Diff", dimension_difference[4], dimension_difference[6], dimension_difference[5], dimension_difference[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "acc", "Rank", dimension_ranks[4], dimension_ranks[6], dimension_ranks[5], dimension_ranks[7]]))

    writer = pd.ExcelWriter('Results/acc_results.xlsx')
    df1 = pd.DataFrame(numpy.asarray(ns_alg_1_matrix))
    df2 = pd.DataFrame(numpy.asarray(ns_alg_2_matrix))
    df1.to_excel(writer, 'Sheet1')
    df2.to_excel(writer, 'Sheet2')
    writer.save()
    return


def find_best_results():
    result_matrix = load_results("acc_combinations_raw")
    true_pof_hv = load_results("true_pof_hv")
    # we need to tally wins and losses for each
    ns_alg_1_matrix = list()
    ns_alg_1_matrix.append(numpy.asarray(["Algorithm", "Archive Strategy", "Dimension", "Parameter Combination", "Accuracy", "Stability"]))
    # [algorithm, archive_strat, dimension, param_comb]
    best_indexes = [-1, -1, -1, -1]
    best_values = [float('inf'), float('inf')]
    for algorithm in range(len(result_matrix)):
        for archive_strategy in range(len(result_matrix[algorithm])):
            for dimension in range(len(result_matrix[algorithm][archive_strategy])):
                for parameter_combination in range(len(result_matrix[algorithm][archive_strategy][dimension])):
                    parameter_combination_average_hvd = list()
                    parameter_combination_average_stab = list()
                    for benchmark in range(len(result_matrix[algorithm][archive_strategy][dimension][parameter_combination])):
                        benchmark_average_hvd = list()
                        benchmark_average_stab = list()
                        for run in range(len(result_matrix[algorithm][archive_strategy][dimension][parameter_combination][benchmark])):
                            hvd_list = list()
                            stab_list = list()
                            for archive_hv in range(1, len(result_matrix[algorithm][archive_strategy][dimension][parameter_combination][benchmark][run])):
                                # we need to find which algorithm/strategy/parameter_combination produces the best pof's for the average run overall benchmark functions
                                # get the performance measure for the archive
                                # calculate hvd
                                cur_pof_volume = result_matrix[algorithm][archive_strategy][dimension][parameter_combination][benchmark][run][archive_hv]
                                prev_pof_volume = result_matrix[algorithm][archive_strategy][dimension][parameter_combination][benchmark][run][archive_hv-1]
                                true_pof_volume = true_pof_hv[parameter_combination][benchmark][archive_hv % len(true_pof_hv[parameter_combination][benchmark])]
                                current_hvd = math.fabs(true_pof_volume - cur_pof_volume)
                                prev_hvd = math.fabs(true_pof_volume - prev_pof_volume)
                                # calculate stab
                                stab = max(0, (prev_hvd - current_hvd))
                                # Add the hvd and the stab to a list
                                hvd_list.append(current_hvd)
                                stab_list.append(stab)
                            # add the average hvd for the first archive
                            hvd_list.append(true_pof_hv[parameter_combination][benchmark][0] - result_matrix[algorithm][archive_strategy][dimension][parameter_combination][benchmark][run][0])
                            # add the average for this hvd and stab to the benchmark average lists
                            benchmark_average_hvd.append(numpy.mean(hvd_list))
                            if math.isnan(numpy.mean(stab_list)):
                                benchmark_average_stab.append(0)
                            else:
                                benchmark_average_stab.append(numpy.mean(stab_list))
                        # reduce the list to averages
                        benchmark_average_hvd_value = numpy.mean(benchmark_average_hvd)
                        if math.isnan(numpy.mean(numpy.mean(benchmark_average_stab))):
                            benchmark_average_stab_value = 0
                        else:
                            benchmark_average_stab_value = numpy.mean(benchmark_average_stab)
                        parameter_combination_average_hvd.append(benchmark_average_hvd_value)
                        parameter_combination_average_stab.append(benchmark_average_stab_value)
                    # we need to store the average of all benchmark values
                    parameter_combination_average_hvd_value = numpy.mean(parameter_combination_average_hvd)
                    if math.isnan(numpy.mean(parameter_combination_average_stab)):
                        parameter_combination_average_stab_value = 0
                    else:
                        parameter_combination_average_stab_value = numpy.mean(parameter_combination_average_stab)
                    if parameter_combination_average_hvd_value < best_values[0] and parameter_combination_average_stab_value < best_values[1]:
                        best_values[0] = parameter_combination_average_hvd_value
                        best_values[1] = parameter_combination_average_stab_value

                        best_indexes = [algorithm, archive_strategy, dimension, parameter_combination]
                    # ns_alg_1_matrix.append(numpy.asarray(print_results(algorithm, archive_strategy, dimension, parameter_combination, parameter_combination_average_hvd_value, parameter_combination_average_stab_value)))
                    ns_alg_1_matrix.append(numpy.asarray([algorithm, archive_strategy, dimension, parameter_combination, parameter_combination_average_hvd_value, parameter_combination_average_stab_value]))
    # sort by the last two columns

    print("Best Results")
    # Remeber that the results for archive strategy 2 and 3 are swapped in the report (for as in archive strategy 2 in the results is actually archive strategy 3 in the report)
    print("algorithm: "+str(best_indexes[0]) + ", archive_strategy: "+str(best_indexes[1]) + ", dimension: "+str(best_indexes[2]) + ", parameter combination: "+str(best_indexes[3]))
    writer = pd.ExcelWriter('Results/combination_results.xlsx')
    df1 = pd.DataFrame(numpy.asarray(ns_alg_1_matrix))
    df1.to_excel(writer, 'Sheet1')
    writer.save()
    return


def read_stab(directory):
    print("reading stab")
    # load the reference vectors
    ref_x = load_results("x_max_reference")
    ref_y = load_results("y_max_reference")

    # load the TRUE POF's for each benchmark function
    true_pof_hv = load_results("true_pof_hv")
    hv_matrix = load_results("acc_combinations_raw")

    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    algorithm_values = list()
    algorithm_combination = list()
    for algotihm_folder in range(len(hv_matrix)):
        print("algotihm_folder: " + str(algotihm_folder))
        archive_strategy_index = -1
        archive_strategy_average_values = list()
        archive_strategy_average_combination = list()
        for archive_strategy in range(len(hv_matrix[algotihm_folder])):
            print("archive_strategy: " + str(archive_strategy))
            archive_strategy_index += 1
            dimension_average_values = list()
            dimension_average_combination = list()
            for dimension in range(len(hv_matrix[algotihm_folder][archive_strategy])):
                print("dimension: " + str(dimension))
                combination_average_values = list()
                combination_average_combination = list()
                for param_combination in range(len(hv_matrix[algotihm_folder][archive_strategy][dimension])):
                    print("param_combination: " + str(param_combination))
                    benchmark_average_values = list()
                    benchmark_average_combination = list()
                    for benchmark in range(len(hv_matrix[algotihm_folder][archive_strategy][dimension][param_combination])):
                        print("benchmark: " + str(benchmark))
                        run_average_list = list()
                        run_list_combinations = list()
                        for run in range(len(hv_matrix[algotihm_folder][archive_strategy][dimension][param_combination][benchmark])):
                            archive_average_list = list()
                            for archive_hv in range(1, len(hv_matrix[algotihm_folder][archive_strategy][dimension][param_combination][benchmark][run])):

                                true_pof_volume = true_pof_hv[param_combination][benchmark][archive_hv % len(true_pof_hv[param_combination][benchmark])]

                                cur_hvd = math.fabs(true_pof_volume - hv_matrix[algotihm_folder][archive_strategy][dimension][param_combination][benchmark][run][archive_hv])
                                prev_hvd = math.fabs(true_pof_volume - hv_matrix[algotihm_folder][archive_strategy][dimension][param_combination][benchmark][run][archive_hv-1])
                                stab = max(0, (prev_hvd - cur_hvd))
                                archive_average_list.append(stab)

                            if archive_average_list:
                                # calculate the average ns for all archives of this run
                                run_average = sum(archive_average_list) / len(archive_average_list)
                                # add to the list of all runs for this benchmark
                                run_average_list.append(run_average)
                                run_list_combinations.append(archive_average_list)
                        benchmark_average = sum(run_average_list) / len(run_average_list)
                        benchmark_average_values.append(benchmark_average)
                        benchmark_average_combination.append(run_list_combinations)
                    combination_average_values.append(benchmark_average_values)
                    combination_average_combination.append(benchmark_average_combination)
                dimension_average_values.append(combination_average_values)
                dimension_average_combination.append(combination_average_combination)
            archive_strategy_average_values.append(dimension_average_values)
            archive_strategy_average_combination.append(dimension_average_combination)
        algorithm_values.append(archive_strategy_average_values)
        algorithm_combination.append(archive_strategy_average_combination)
    pickle_results(algorithm_values, "stab_raw")
    pickle_results(algorithm_combination, "stab_combinations_raw")
    return


def stats_stab():
    result_matrix = load_results("stab_raw")
    # we need to tally wins and losses for each
    ns_alg_1_matrix = list()
    ns_alg_2_matrix = list()
    ns_alg_1_matrix.append(numpy.asarray(["D", "PM", "Results", "A1", "A2", "A3", "A4"]))
    ns_alg_2_matrix.append(numpy.asarray(["D", "PM", "Results", "A1", "A2", "A3", "A4"]))
    for dimension in range(len(result_matrix[0][0])):
        dimension_wins = [0, 0, 0, 0, 0, 0, 0, 0]
        dimension_losses = [0, 0, 0, 0, 0, 0, 0, 0]
        # calculate the wins/losses/difference/rank for each strategy
        # for each parameter combination
        for param_combination in range(len(result_matrix[0][0][dimension])):
            # for each benchmark
            for benchmark in range(len(result_matrix[0][0][dimension][param_combination])):
                # for both algorithms
                alg_index = 0
                for algorithm_index in range(len(result_matrix)):
                    # identify which archive management strategy wins for this benchmark
                    # get dimension name
                    if dimension == 0:
                        dimension_name = "Large"
                    elif dimension == 1:
                        dimension_name = "Low"
                    elif dimension == 2:
                        dimension_name = "Medium"
                    largest_ns = -1
                    largest_ns_index = -1
                    # for each archive management strategy
                    for archive_strategy in range(len(result_matrix[algorithm_index])):
                        # find the largest ns for this parameter/benchmark combination
                        current_benchmark = result_matrix[algorithm_index][archive_strategy][dimension][param_combination][benchmark]
                        if current_benchmark > largest_ns:
                            largest_ns = current_benchmark
                            largest_ns_index = archive_strategy
                    # allocate the winning strategy a win and the others a loss
                    for archive_strategy in range(len(result_matrix[algorithm_index])):
                        if archive_strategy == largest_ns_index:
                            # allocate a win
                            dimension_wins[archive_strategy+alg_index] += 1
                        else:
                            # allocate a loss
                            dimension_losses[archive_strategy+alg_index] += 1
                    alg_index += 4

        # create the difference
        dimension_difference = list()
        for value_index in range(len(dimension_wins)):
            dimension_difference.append(dimension_wins[value_index]-dimension_losses[value_index])
        # create the ranked list
        dimension_ranks = ss.rankdata(numpy.negative(dimension_wins), method='min')

        # save to algorithm 1 matrix
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "stab", "Wins", dimension_wins[0], dimension_wins[2], dimension_wins[1], dimension_wins[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "stab", "Losses", dimension_losses[0], dimension_losses[2], dimension_losses[1], dimension_losses[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "stab", "Diff", dimension_difference[0], dimension_difference[2], dimension_difference[1], dimension_difference[3]]))
        ns_alg_1_matrix.append(numpy.asarray([dimension_name, "stab", "Rank", dimension_ranks[0], dimension_ranks[2], dimension_ranks[1], dimension_ranks[3]]))

        # save to algorithm 2 matrix
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "stab", "Wins", dimension_wins[4], dimension_wins[6], dimension_wins[5], dimension_wins[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "stab", "Losses", dimension_losses[4], dimension_losses[6], dimension_losses[5], dimension_losses[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "stab", "Diff", dimension_difference[4], dimension_difference[6], dimension_difference[5], dimension_difference[7]]))
        ns_alg_2_matrix.append(numpy.asarray([dimension_name, "stab", "Rank", dimension_ranks[4], dimension_ranks[6], dimension_ranks[5], dimension_ranks[7]]))

    writer = pd.ExcelWriter('Results/stab_results.xlsx')
    df1 = pd.DataFrame(numpy.asarray(ns_alg_1_matrix))
    df2 = pd.DataFrame(numpy.asarray(ns_alg_2_matrix))
    df1.to_excel(writer, 'Sheet1')
    df2.to_excel(writer, 'Sheet2')
    writer.save()
    return


def read_acc_combinations(directory):
    print("reading acc combinations")
    # load the reference vectors
    ref_x = load_results("x_max_reference")
    ref_y = load_results("y_max_reference")

    # load the TRUE POF's for each benchmark function
    true_pof = load_results("true_pof")

    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    algorithm_values = list()
    for algotihm_folder in algorithm_folders:
        algotihm_folder_directory = directory + "/" + algotihm_folder
        archive_strategies = [file for file in listdir(algotihm_folder_directory) if isfile(join(algotihm_folder_directory, file))]
        archive_strategy_index = -1
        archive_strategy_average_values = list()
        for archive_strategy in archive_strategies:
            print("archive_strategy: " + str(archive_strategy))
            archive_strategy_index += 1
            result_matrix = load_results(algotihm_folder_directory + "/" + archive_strategy)
            dimension_average_values = list()
            for dimension in range(len(result_matrix)):
                combination_average_values = list()
                for param_combination in range(len(result_matrix[dimension])):
                    benchmark_average_values = list()
                    for benchmark in range(len(result_matrix[dimension][param_combination])):
                        run_average_list = list()
                        for run in range(len(result_matrix[dimension][param_combination][benchmark][0])):
                            archive_average_list = list()
                            for archive in range(len(result_matrix[dimension][param_combination][benchmark][0][run])):
                                # we are at an archive
                                archive_x = list(map(float, result_matrix[dimension][param_combination][benchmark][0][run][archive]))
                                archive_y = list(map(float, result_matrix[dimension][param_combination][benchmark][1][run][archive]))
                                true_archive_x = list(map(float, true_pof[param_combination][benchmark][archive % len(true_pof[param_combination][benchmark])][1]))
                                true_archive_y = list(map(float, true_pof[param_combination][benchmark][archive % len(true_pof[param_combination][benchmark])][0]))

                                if max(archive_x) == 0 or max(archive_y) == 0 or max(true_archive_y) == 0:
                                    continue

                                # calculate the HVD for each pof
                                # get the reference point for this benchmark function
                                referencePoint = [ref_x[benchmark], ref_y[benchmark]]
                                # print("reference point")
                                # print(referencePoint)
                                hv = HyperVolume.HyperVolume(referencePoint)
                                # calculate the HV of the current archive
                                # convert the archive arrays into a front
                                front = list()
                                for point_x, point_y in zip(archive_x, archive_y):
                                    # make sure the points do not already exist in the front
                                    if not [point_x, point_y] in front:
                                        front.append([point_x, point_y])

                                # print("front, archive_strategy: "+str(archive_strategy) + ", dimension: "+str(dimension)+
                                #       ", param_combination: "+str(param_combination) + ", benchmark: "+str(benchmark)+", run: "+str(run)+", archive: "+str(archive))
                                # print(front)
                                pof_volume = hv.compute(front)

                                # calculate the HV of the true POF for the param_combination/benchmark function
                                # convert the archive arrays into a front
                                front = list()
                                check = True

                                for point_x, point_y in zip(true_archive_x, true_archive_y):
                                    # check if the point_y is not already in the front
                                    for elem in front:
                                        if round(point_y, 14) == elem[1]:
                                            check = False
                                    if check:
                                        front.append([point_x, point_y])
                                    check = True
                                true_pof_volume = hv.compute(front)
                                hvd = math.fabs(true_pof_volume - pof_volume)
                                archive_average_list.append(hvd)

                            # calculate the average ns for all archives of this run
                            # if archive_average_list:
                            #     run_average = sum(archive_average_list) / len(archive_average_list)
                            #     # add to the list of all runs for this benchmark
                            run_average_list.append(archive_average_list)
                        # benchmark_average = sum(run_average_list) / len(run_average_list)
                        benchmark_average_values.append(run_average_list)
                    combination_average_values.append(benchmark_average_values)
                dimension_average_values.append(combination_average_values)
            archive_strategy_average_values.append(dimension_average_values)
        algorithm_values.append(archive_strategy_average_values)
    pickle_results(algorithm_values, "acc_combinations_raw")
    return


def read_stab_combinations(directory):
    print("reading stab combinations")
    # load the reference vectors
    ref_x = load_results("x_max_reference")
    ref_y = load_results("y_max_reference")

    # load the TRUE POF's for each benchmark function
    true_pof = load_results("true_pof")

    algorithm_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    algorithm_values = list()
    for algotihm_folder in algorithm_folders:
        algotihm_folder_directory = directory + "/" + algotihm_folder
        archive_strategies = [file for file in listdir(algotihm_folder_directory) if isfile(join(algotihm_folder_directory, file))]
        archive_strategy_index = -1
        archive_strategy_average_values = list()
        for archive_strategy in archive_strategies:
            print("archive_strategy: "+str(archive_strategy))
            archive_strategy_index += 1
            result_matrix = load_results(algotihm_folder_directory + "/" + archive_strategy)
            dimension_average_values = list()
            for dimension in range(len(result_matrix)):
                combination_average_values = list()
                for param_combination in range(len(result_matrix[dimension])):
                    benchmark_average_values = list()
                    for benchmark in range(len(result_matrix[dimension][param_combination])):
                        run_average_list = list()
                        for run in range(len(result_matrix[dimension][param_combination][benchmark][0])):
                            archive_average_list = list()
                            for archive in range(len(result_matrix[dimension][param_combination][benchmark][0][run])):
                                # we are at an archive
                                archive_x = list(map(float, result_matrix[dimension][param_combination][benchmark][0][run][archive]))
                                archive_y = list(map(float, result_matrix[dimension][param_combination][benchmark][1][run][archive]))
                                prev_archive_x = list(map(float, result_matrix[dimension][param_combination][benchmark][0][run][archive-1]))
                                prev_archive_y = list(map(float, result_matrix[dimension][param_combination][benchmark][1][run][archive-1]))
                                true_archive_x = list(map(float, true_pof[param_combination][benchmark][archive % len(true_pof[param_combination][benchmark])][1]))
                                true_archive_y = list(map(float, true_pof[param_combination][benchmark][archive % len(true_pof[param_combination][benchmark])][0]))

                                if max(archive_x) == 0 or max(archive_y) == 0 or max(prev_archive_x) == 0 or max(prev_archive_y) == 0 or max(true_archive_y) == 0:
                                    continue

                                # calculate the HVD for each pof
                                # get the reference point for this benchmark function
                                referencePoint = [ref_x[benchmark], ref_y[benchmark]]
                                hv = HyperVolume.HyperVolume(referencePoint)

                                # convert the archive arrays into a front
                                front = list()
                                for point_x, point_y in zip(archive_x, archive_y):
                                    if not [point_x, point_y] in front:
                                        front.append([point_x, point_y])

                                # calculate the HV of the current archive
                                cur_pof_volume = hv.compute(front)

                                # convert the archive arrays into a front
                                front = list()
                                for point_x, point_y in zip(prev_archive_x, prev_archive_y):
                                    if not [point_x, point_y] in front:
                                        front.append([point_x, point_y])

                                # calculate the HV of the current archive
                                prev_pof_volume = hv.compute(front)

                                # calculate the HV of the true POF for the param_combination/benchmark function
                                # convert the archive arrays into a front
                                front = list()
                                check = True
                                for point_x, point_y in zip(true_archive_x, true_archive_y):
                                    # check if the point_y is not already in the front
                                    for elem in front:
                                        if round(point_y, 14) == elem[1]:
                                            check = False
                                    if check:
                                        front.append([point_x, point_y])
                                    check = True
                                true_pof_volume = hv.compute(front)

                                cur_hvd = math.fabs(true_pof_volume - cur_pof_volume)
                                prev_hvd = math.fabs(true_pof_volume - prev_pof_volume)
                                stab = max(0, (prev_hvd - cur_hvd))
                                archive_average_list.append(stab)

                            # if archive_average_list:
                            #     # calculate the average ns for all archives of this run
                            #     run_average = sum(archive_average_list) / len(archive_average_list)
                            #     # add to the list of all runs for this benchmark
                            run_average_list.append(archive_average_list)
                        # benchmark_average = sum(run_average_list) / len(run_average_list)
                        benchmark_average_values.append(run_average_list)
                    combination_average_values.append(benchmark_average_values)
                dimension_average_values.append(combination_average_values)
            archive_strategy_average_values.append(dimension_average_values)
        algorithm_values.append(archive_strategy_average_values)
    pickle_results(algorithm_values, "stab_raw")
    return

# --------------------- HELPER METHODS --------------------- #


def print_results(algorithm, archive_strategy, dimension, parameter_combination, parameter_combination_average_hvd_value, parameter_combination_average_stab_value):
    return_list = list()
    algorithm_name = None
    dimension_name = None
    parameter_combination_name = None
    if algorithm == 0:
        algorithm_name = "D-MGPSO"
    elif algorithm == 1:
        algorithm_name = "D-QMGPSO"

    if dimension == 0:
        dimension_name = "Large"
    elif dimension == 1:
        dimension_name = "Low"
    elif dimension == 2:
        dimension_name = "Medium"

    if parameter_combination == 0:
        parameter_combination_name = "nT_1_tT_10"
    elif parameter_combination == 1:
        parameter_combination_name = "nT_1_tT_25"
    elif parameter_combination == 2:
        parameter_combination_name = "nT_1_tT_50"
    elif parameter_combination == 3:
        parameter_combination_name = "nT_10_tT_10"
    elif parameter_combination == 4:
        parameter_combination_name = "nT_10_tT_25"
    elif parameter_combination == 5:
        parameter_combination_name = "nT_10_tT_50"
    elif parameter_combination == 6:
        parameter_combination_name = "nT_20_tT_10"
    elif parameter_combination == 7:
        parameter_combination_name = "nT_20_tT_25"
    elif parameter_combination == 8:
        parameter_combination_name = "nT_20_tT_50"

    return [algorithm_name, archive_strategy, dimension_name, parameter_combination_name, parameter_combination_average_hvd_value, parameter_combination_average_stab_value]


def get_true_pof_hv(directory):
    true_pof = load_results("true_pof")
    ref_x = load_results("x_max_reference")
    ref_y = load_results("y_max_reference")

    combination_average_values = list()
    for param_combination in range(len(true_pof)):
        print("param_combination: " + str(param_combination))
        benchmark_average_values = list()
        for benchmark in range(len(true_pof[param_combination])):
            print("benchmark: " + str(benchmark))
            archive_average_list = list()
            for archive in range(len(true_pof[param_combination][benchmark])):
                # print("archive: " + str(archive))
                # we are at an archive
                true_archive_x = list(map(float, true_pof[param_combination][benchmark][archive % len(true_pof[param_combination][benchmark])][1]))
                true_archive_y = list(map(float, true_pof[param_combination][benchmark][archive % len(true_pof[param_combination][benchmark])][0]))

                if max(true_archive_y) == 0:
                    continue

                referencePoint = [ref_x[benchmark], ref_y[benchmark]]

                hv = HyperVolume.HyperVolume(referencePoint)

                front = list()
                check = True

                for point_x, point_y in zip(true_archive_x, true_archive_y):
                    # check if the point_y is not already in the front
                    for elem in front:
                        if round(point_y, 14) == elem[1]:
                            check = False
                    if check:
                        front.append([point_x, point_y])
                    check = True
                true_pof_volume = hv.compute(front)
                archive_average_list.append(true_pof_volume)

            if archive_average_list:
                benchmark_average_values.append(archive_average_list)
        combination_average_values.append(benchmark_average_values)
    # pickle and return
    pickle_results(combination_average_values, "true_pof_hv")
    return


def combination_sort(folder_list):
    return [folder_list[3], folder_list[4], folder_list[5], folder_list[0], folder_list[1], folder_list[2], folder_list[6], folder_list[7], folder_list[8]]


def benchmark_sort(folder_list):
    return [folder_list[0], folder_list[1], folder_list[14], folder_list[15],
            folder_list[16], folder_list[17], folder_list[18], folder_list[19],
            folder_list[20], folder_list[21], folder_list[22], folder_list[23],
            folder_list[24], folder_list[25], folder_list[26], folder_list[27],
            folder_list[28], folder_list[29], folder_list[30], folder_list[31],
            folder_list[2], folder_list[3], folder_list[4], folder_list[5],
            folder_list[6], folder_list[7], folder_list[8], folder_list[9],
            folder_list[10], folder_list[11], folder_list[12], folder_list[13]]


def benchmark_sort_2(folder_list):
    return [folder_list[0], folder_list[1], folder_list[8], folder_list[9],
            folder_list[10], folder_list[11], folder_list[12], folder_list[13],
            folder_list[14], folder_list[15], folder_list[2], folder_list[3],
            folder_list[4], folder_list[5], folder_list[6], folder_list[7]]


def run_sort(folder_list):
    return [folder_list[0], folder_list[1], folder_list[12], folder_list[23],
            folder_list[24], folder_list[25], folder_list[26], folder_list[27],
            folder_list[28], folder_list[29], folder_list[2], folder_list[3],
            folder_list[4], folder_list[5], folder_list[6], folder_list[7],
            folder_list[8], folder_list[9], folder_list[10], folder_list[11],
            folder_list[13], folder_list[14], folder_list[15], folder_list[16],
            folder_list[17], folder_list[18], folder_list[19], folder_list[20],
            folder_list[21], folder_list[22]]


def pickle_results(result_matrix, name):
    # fname = name+"_result_matrix.pkl"
    # pickle dump the list object
    with open(name, "wb") as fout:
        # default protocol is zero
        # -1 gives highest prototcol and smallest data file size
        pickle.dump(result_matrix, fout, protocol=-1)


def load_results(fname="result_matrix.pkl"):
    with open(fname, "rb") as fin:
        result_matrix = pickle.load(fin)
    return result_matrix

# --------------------- Performance METHODS --------------------- #

# collect_data("../Dynamic POF")

# find_reference_vectors("../Dynamic POF")
# read_ns("../Dynamic POF")
# stats_ns()
# save_true_pof("Dynamic True POF")
# read_acc("../Dynamic POF")
stats_acc()
# read_stab("../Dynamic POF")
# stats_stab()
# get_true_pof_hv("../Dynamic POF")
# collect_data_specific("../Dynamic POF", ["MGPSO"], ["Archive Strategy 3_result_matrix.pkl"], ["Medium Dimensions"], [2], ["nT_1_tT_25"], [1], [11])

# load_results("../Dynamic POF/MGPSO/Archive Strategy 2_result_matrix.pkl")
# find_best_results()
