import math

# Import classes
from MGPSO import EvaluationsDynamic

import matplotlib.pyplot as plt

# Import weights and biases
# import wandb
# name = "test name"
# wandb.init(name=name, project="MGPSO-Test", id="test")
# print(wandb.name)

def plotTest(pof, pos, name="default", y_limit=[0, 1]):
  fig, ax = plt.subplots()
  ax.set_ylim(y_limit)
  # Plot pos
  if(len(pos) > 1):
    for g_i in pos:
      x, y = [g_i["f1"][0], g_i["f1"][len(g_i["f1"])-1]], [g_i["g"][0], g_i["g"][0]]
      ax.plot(x, y)
  else:
    # plot single g_i vs x
    print("to do")
  plt.show()

  # Plot pof
  plt.figure()
  if(len(pof) > 1):
    # plot multiple pof's
    print("to do")
  else:
    x, y = pof[0]["f1"], pof[0]["f2"]
    ax.plot(x, y)
  plt.show()
  # for g_i in g:
  #     f_plot.append([f1[0], f1[len(f1)-1]])
  #     g_plot.append([g_i, g_i])
  # print("x_axis: ")
  # print(f_plot)
  # print("y_axis: ")
  # print(g_plot)
  # plt.plot(f_plot, g_plot, 'r')
  # plt.ylabel('some numbers')
  # plt.show()
  # return pof
  # wandb.save("Main.py")
  # self.__save_true_pof(pof, 0)

def main():
  evaluations_dynamic = EvaluationsDynamic.EvaluationsDynamic()

  # --- DIMP2 --- #
  # evaluations_dynamic.set_severity_of_change(10)
  # evaluations_dynamic.set_frequency_of_change(10)

  # evaluations_dynamic.dimp2()
  # evaluations_dynamic.dimp2_generate_pof(1000)

  # --- fda1 --- #
  # evaluations_dynamic.set_severity_of_change(10)
  # evaluations_dynamic.set_frequency_of_change(10)

  # evaluations_dynamic.fda1()
  pof = evaluations_dynamic.fda1_generate_pof(1000)
  pos = evaluations_dynamic.fda1_generate_pos(1000)
  plotTest(pof, pos)

main()
