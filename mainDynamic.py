import PSODynamic
import EvaluationsDynamic


def main():
    evaluations_dynamic = EvaluationsDynamic.EvaluationsDynamic()
    severity_of_change = [1, 10, 20]
    frequency_of_change = [10, 25, 50]

    for run in range(1):
        evaluations_dynamic.set_severity_of_change(severity_of_change[1])
        evaluations_dynamic.set_frequency_of_change(frequency_of_change[1])
        evaluations_dynamic.set_run(run)
        # -- Test DIMP2 -- #
        evaluations_dynamic.fda2()
        PSODynamic.PSODynamic(1000, evaluations_dynamic)

main()
