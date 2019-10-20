import math

# Import classes
from MGPSO import EvaluationsDynamic

import matplotlib.pyplot as plt

# Import weights and biases
# import wandb
# name = "test name"
# wandb.init(name=name, project="MGPSO-Test", id="test")
# print(wandb.name)

def plotTest(pof, pos, name, x_name1, x_name2, y_name1, y_name2, y_limit=[0, 1]):
  #  Set Figure size
  plt.figure(figsize=(10,4))

  # Plot POS
  ax = plt.subplot(1, 2, 1)
  plt.title('POF & POS: '+name)
  plt.ylabel('G(t)')
  plt.xlabel('x')
  ax.set_ylim(y_limit)
  for pos_i in pos:
    ax.plot(pos_i["f1"], pos_i["g"])
  
  # Plot POF
  ax2 = plt.subplot(1, 2, 2)
  plt.ylabel('f2')
  plt.xlabel('f1')
  for pof_i in pof:
    x, y = pof_i["f1"], pof_i["f2"]
    ax2.plot(x, y)
  plt.show()
  return

def truPof():

  evaluations_dynamic = EvaluationsDynamic.EvaluationsDynamic()
  evaluations_dynamic.set_severity_of_change(10)
  evaluations_dynamic.set_frequency_of_change(10)

  # # --- DIMP2 --- #
  # pof = evaluations_dynamic.dimp2_generate_pof(1000)
  # pos = evaluations_dynamic.dimp2_generate_pos(1000)
  # plotTest(pof, pos, "dimp2", 'f1', "f1","G(t)", "f2")

  # # --- fda1 --- #
  # pof = evaluations_dynamic.fda1_generate_pof(1000)
  # pos = evaluations_dynamic.fda1_generate_pos(1000, 1000)
  # plotTest(pof, pos, "fda1", 'f1', "f1","G(t)", "f2")

  # # --- fda1_zhou (zjz) --- #
  # pof = evaluations_dynamic.fda1_zhou_generate_pof(1000)
  # pos = evaluations_dynamic.fda1_zhou_generate_pos(1000, 1000)
  # plotTest(pof, pos, "zjz", 'f1', "f1","G(t) + x1^H(t)", "f2")

truPof()
