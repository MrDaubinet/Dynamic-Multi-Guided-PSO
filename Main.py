# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import PSO
import Evaluations


def main():
    evaluations = Evaluations.Evaluations()
    # -- Test zdt1 -- #
    # print("zdt1")
    # evaluations.zdt1()
    # -- Test zdt2 -- #
    # print("zdt2")
    # evaluations.zdt2()
    # -- Test zdt3 -- #
    # print("zdt3")
    # evaluations.zdt3()
    # -- Test zdt4 -- #
    # evaluations.zdt4()
    # print("zdt4")
    # -- Test zdt6 -- #
    # evaluations.zdt6()
    # print("zdt6")

    # -- Test wfg1_2obj -- #
    # evaluations.wfg1_2obj()
    # print("wfg1_2obj")

    # -- Test wfg2_2obj -- #
    # evaluations.wfg2_2obj()
    # print("wfg2_2obj")

    # -- Test wfg3_2obj -- #
    # evaluations.wfg3_2obj()
    # print("wfg3_2obj")

    # -- Test wfg4_2obj -- #
    # evaluations.wfg4_2obj()
    # print("wfg4_2obj")

    # -- Test wfg5_2obj -- #
    # evaluations.wfg5_2obj()
    # print("wfg5_2obj")

    # -- Test wfg6_2obj -- #
    # evaluations.wfg6_2obj()
    # print("wfg6_2obj")

    # -- Test wfg7_2obj -- #
    # evaluations.wfg7_2obj()
    # print("wfg7_2obj")

    # -- Test wfg8_2obj -- #
    # evaluations.wfg8_2obj()
    # print("wfg8_2obj")

    # -- Test wfg9_2obj -- #
    evaluations.wfg9_2obj()
    print("wfg9_2obj")

    # -- Test wfg1_3obj -- #
    # evaluations.wfg1_3obj()
    # print("wfg1_3obj")

    PSO.PSO(2000, evaluations)

main()
