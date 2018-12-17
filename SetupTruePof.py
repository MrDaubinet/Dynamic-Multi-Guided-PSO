import EvaluationsDynamicLowDimensions
from multiprocessing.dummy import Pool as ThreadPool


class SetupTruePof:
    def __init__(self):
        SetupTruePof.set_start_threads()
        # self.set_test_start_threads()

    @staticmethod
    def set_start_threads():
        nt_list = []
        tt_list = []
        benchmark_list = []
        for benchmark in range(16):
            for nT in [0, 1, 2]:
                for tT in [0, 1, 2]:
                    nt_list.append(nT)
                    tt_list.append(tT)
                    benchmark_list.append(benchmark)

        pool = ThreadPool()
        pool.starmap(SetupTruePof.process_function, zip(nt_list, tt_list, benchmark_list))
        pool.close()
        pool.join()

    @staticmethod
    def process_function(nT, tT, benchmark_index):
        print("nT: "+str(nT)+" tT: "+str(tT)+" benchmark: "+str(benchmark_index))
        evaluations_dynamic = EvaluationsDynamicLowDimensions.EvaluationsDynamic()
        benchmarks = [
            evaluations_dynamic.dimp2, evaluations_dynamic.fda1, evaluations_dynamic.fda1_zhou, evaluations_dynamic.fda2, evaluations_dynamic.fda2_camara, evaluations_dynamic.fda3,
            evaluations_dynamic.fda3_camara, evaluations_dynamic.dmop2, evaluations_dynamic.dmop3, evaluations_dynamic.dmop2_iso, evaluations_dynamic.dmop2_dec, evaluations_dynamic.he_1,
            evaluations_dynamic.he_2, evaluations_dynamic.he_6, evaluations_dynamic.he_7, evaluations_dynamic.he_9]

        benchmark_pof = [
            evaluations_dynamic.dimp2_generate_pof, evaluations_dynamic.fda1_generate_pof, evaluations_dynamic.fda1_zhou_generate_pof, evaluations_dynamic.fda2_generate_pof,
            evaluations_dynamic.fda2_camara_generate_pof, evaluations_dynamic.fda3_generate_pof, evaluations_dynamic.fda3_camara_generate_pof, evaluations_dynamic.dmop2_generate_pof,
            evaluations_dynamic.dmop3_generate_pof, evaluations_dynamic.dmop2_iso_generate_pof, evaluations_dynamic.dmop2_dec_generate_pof, evaluations_dynamic.he_1_generate_pof,
            evaluations_dynamic.he_2_generate_pof, evaluations_dynamic.he_6_generate_pof, evaluations_dynamic.he_7_generate_pof, evaluations_dynamic.he_9_generate_pof]

        severity_of_change = [1, 10, 20]
        frequency_of_change = [10, 25, 50]

        evaluations_dynamic.set_severity_of_change(severity_of_change[nT])
        evaluations_dynamic.set_frequency_of_change(frequency_of_change[tT])
        benchmarks[benchmark_index]()
        benchmark_pof[benchmark_index](1001)

    def set_test_start_threads(self):
        nt_list = []
        tt_list = []
        benchmark_list = []
        for benchmark in range(7, 10):
            nT = 1
            tT = 0
            nt_list.append(nT)
            tt_list.append(tT)
            benchmark_list.append(benchmark)

        # print("Creating true front nT: "+str(nT)+" tT: "+str(tT))
        pool = ThreadPool()
        pool.starmap(SetupTruePof.process_function, zip(nt_list, tt_list, benchmark_list))
        pool.close()
        pool.join()
