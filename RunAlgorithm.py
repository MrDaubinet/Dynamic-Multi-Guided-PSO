import EvaluationsDynamicLowDimensions
import EvaluationsDynamicMediumDimensions
import EvaluationsDynamicLargeDimensions

import MGPSODynamicStrategy1
import MGPSODynamicStrategy2
import MGPSODynamicStrategy3
import MGPSODynamicStrategy4

import QMGPSODynamicStrategy1
import QMGPSODynamicStrategy2
import QMGPSODynamicStrategy3
import QMGPSODynamicStrategy4

from multiprocessing.dummy import Pool as ThreadPool


class RunAlgorithm:
    def __init__(self, algorithm, archive_strategy, dimensions):
        self.set_start_threads(algorithm, archive_strategy, dimensions)

    def set_start_threads(self, algorithm, archive_strategy, dimensions):
        print("Algorithm: "+str(algorithm))
        print("Archive Strategy: "+str(archive_strategy))
        print("Dimension: "+str(dimensions))
        nt_list = []
        tt_list = []
        benchmark_list = []
        run_list = []
        run_number_list = []
        dimensions_type_list = []
        algorithm_list = []
        archive_strategy_list = []

        run_count = 0
        run_total = 10*16*3*3
        print("Total number of runs: "+str(run_total))
        for run in range(0, 10):
            for benchmark in range(16):
                for nT in range(3):
                    for tT in range(3):
                        nt_list.append(nT)
                        tt_list.append(tT)
                        benchmark_list.append(benchmark)
                        algorithm_list.append(algorithm)
                        archive_strategy_list.append(archive_strategy)
                        dimensions_type_list.append(dimensions)
                        run_list.append(run)
                        run_number_list.append(run_count)
                        run_count += 1

        pool = ThreadPool()
        pool.starmap(self.process_function, zip(nt_list, tt_list, benchmark_list, run_list, dimensions_type_list, algorithm_list, archive_strategy_list, run_number_list))
        pool.close()
        pool.join()

    @staticmethod
    def process_function(nT, tT, benchmark_index, run_index, dimensions, algorithm, archive_strategy, run_number_index):
        # Create the evaluations object
        if dimensions == 0:
            evaluations_dynamic = EvaluationsDynamicLowDimensions.EvaluationsDynamic()
            evaluations_dynamic.set_dimensions_type("Low Dimensions")
        elif dimensions == 1:
            evaluations_dynamic = EvaluationsDynamicMediumDimensions.EvaluationsDynamic()
            evaluations_dynamic.set_dimensions_type("Medium Dimensions")
        elif dimensions == 2:
            evaluations_dynamic = EvaluationsDynamicLargeDimensions.EvaluationsDynamic()
            evaluations_dynamic.set_dimensions_type("Large Dimensions")

        # create the benchmark functions from the evaluations object
        benchmarks = [
            evaluations_dynamic.dimp2, evaluations_dynamic.fda1, evaluations_dynamic.fda1_zhou, evaluations_dynamic.fda2, evaluations_dynamic.fda2_camara, evaluations_dynamic.fda3,
            evaluations_dynamic.fda3_camara, evaluations_dynamic.dmop2, evaluations_dynamic.dmop3, evaluations_dynamic.dmop2_iso, evaluations_dynamic.dmop2_dec, evaluations_dynamic.he_1,
            evaluations_dynamic.he_2, evaluations_dynamic.he_3, evaluations_dynamic.he_6, evaluations_dynamic.he_7]

        # create the generate_pof functions from the evaluations object
        benchmark_pof = [
            evaluations_dynamic.dimp2_generate_pof, evaluations_dynamic.fda1_generate_pof, evaluations_dynamic.fda1_zhou_generate_pof, evaluations_dynamic.fda2_generate_pof,
            evaluations_dynamic.fda2_camara_generate_pof, evaluations_dynamic.fda3_generate_pof, evaluations_dynamic.fda3_camara_generate_pof, evaluations_dynamic.dmop2_generate_pof,
            evaluations_dynamic.dmop3_generate_pof, evaluations_dynamic.dmop2_iso_generate_pof, evaluations_dynamic.dmop2_dec_generate_pof, evaluations_dynamic.he_1_generate_pof,
            evaluations_dynamic.he_2_generate_pof, evaluations_dynamic.he_3_generate_pof, evaluations_dynamic.he_6_generate_pof, evaluations_dynamic.he_7_generate_pof]

        # set the optional parameter test values
        severity_of_change = [1, 10, 20]
        frequency_of_change = [10, 25, 50]

        # set evaluation to the correct parameter combination
        evaluations_dynamic.set_severity_of_change(severity_of_change[nT])
        evaluations_dynamic.set_frequency_of_change(frequency_of_change[tT])
        evaluations_dynamic.set_run(run_index)

        print("nT: " + str(severity_of_change[nT]) + " tT: " + str(frequency_of_change[tT]) + " benchmark: " + str(benchmark_index))
        # set the benchmark
        benchmarks[benchmark_index]()

        # run the algorithm
        RunAlgorithm.run_algorithm(evaluations_dynamic, algorithm, archive_strategy)
        print("Run :"+str(run_number_index)+" completed")

    @staticmethod
    def run_algorithm(evaluations_dynamic, algorithm, archive_strategy):
        if algorithm == 0:
            # MGPSO
            if archive_strategy == 0:
                # Archive Strategy 1
                MGPSODynamicStrategy1.PSODynamic(1000, evaluations_dynamic)
                return
            if archive_strategy == 1:
                # Archive Strategy 2
                MGPSODynamicStrategy2.PSODynamic(1000, evaluations_dynamic)
                return
            if archive_strategy == 2:
                # Archive Strategy 3
                MGPSODynamicStrategy3.PSODynamic(1000, evaluations_dynamic)
                return
            if archive_strategy == 3:
                # Archive Strategy 4
                MGPSODynamicStrategy4.PSODynamic(1000, evaluations_dynamic)
                return
        else:
            # Quantum MGPSO
            if archive_strategy == 0:
                # Archive Strategy 1
                QMGPSODynamicStrategy1.PSODynamic(1000, evaluations_dynamic)
                return
            if archive_strategy == 1:
                # Archive Strategy 2
                QMGPSODynamicStrategy2.PSODynamic(1000, evaluations_dynamic)
                return
            if archive_strategy == 2:
                # Archive Strategy 3
                QMGPSODynamicStrategy3.PSODynamic(1000, evaluations_dynamic)
                return
            if archive_strategy == 3:
                # Archive Strategy 4
                QMGPSODynamicStrategy4.PSODynamic(1000, evaluations_dynamic)
                return

    def set_test_start_threads(self, algorithm, archive_strategy, dimensions):
        nt_list = []
        tt_list = []
        benchmark_list = []
        run_list = []
        dimensions_type_list = []
        algorithm_list = []
        archive_strategy_list = []
        run_number_list = []
        run_count = 0

        for run in range(1):
            for benchmark in range(16):
                nT = 1
                tT = 0
                nt_list.append(nT)
                tt_list.append(tT)
                benchmark_list.append(benchmark)
                run_list.append(run)
                algorithm_list.append(algorithm)
                archive_strategy_list.append(archive_strategy)
                dimensions_type_list.append(dimensions)
                run_number_list.append(run_count)
                run_count += 1

        pool = ThreadPool()
        pool.starmap(RunAlgorithm.process_function, zip(nt_list, tt_list, benchmark_list, run_list, dimensions_type_list, algorithm_list, archive_strategy_list, run_number_list))
        pool.close()
        pool.join()
