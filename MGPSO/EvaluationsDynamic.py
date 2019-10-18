import math
import random
# Import weights and biases
# import wandb

class EvaluationsDynamic:
    def __init__(self):
        self.__objectives = []
        self.__objective_types = []
        self.__bounds = []
        self.__num_particles = []
        self.__num_dimensions = None
        self.__objective_names = []
        self.__constants = []
        self.__tournament_selection_size = None
        self.__extra_bounds = None
        self.__current_bench = None
        self.__true_pof = None
        self.__severity_of_change = None
        self.__frequency_of_change = None
        self.__r_i = None
        self.__t = None
        self.__run_number = None

    # ------------------------------------------------------------------------------------------------- Class Functions ----------------------------------------------------------------------------------------------------- #
    def get_objective_functions(self):
        return self.__objectives

    def get_objective_names(self):
        return self.__objective_names

    def num_objectives(self):
        return len(self.__objectives)

    def get_constants(self):
        return self.__constants

    def get_objective_types(self):
        return self.__objective_types

    def get_bounds(self):
        return self.__bounds

    def get_num_particles(self):
        return self.__num_particles

    def get_num_dimensions(self):
        return self.__num_dimensions

    def get_tournament_selection_size(self):
        return self.__tournament_selection_size

    def get_extra_bounds(self):
        return self.__extra_bounds

    def get_true_pof(self):
        if self.__true_pof is None:
            self.__true_pof = []
            # extract the true pof from the textfile
            with open("True POF/" + self.__current_bench + ".pf") as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            for item in content:
                self.__true_pof.append(list(map(float, item.split())))
        return self.__true_pof

    def set_severity_of_change(self, s):
        self.__severity_of_change = s
        wandb.config.severity_of_change = s

    def set_frequency_of_change(self, f):
        self.__frequency_of_change = f
        wandb.config.frequency_of_change = f

    def update_t(self, iteration):
        self.__t = (1/self.__severity_of_change) * \
            math.floor(iteration/self.__frequency_of_change)
        return

    def get_t(self):
        return self.__t

    def set_run(self, run):
        self.__run_number = run
        return

    def get_run(self):
        return self.__run_number

    def get_r_i(self):
        return self.__r_i

    # ----------------------------------------------- DIMP2 ------------------------------------------------------------ #
    @staticmethod
    def __dimp2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dimp2_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __dimp2_g(x, t):
        sum = 0
        x_ii = x[1:len(x)]
        for i, x_i in enumerate(x_ii, start=0):
            # for x_i in range(0, len(x_ii)):
            sum += math.pow((x_i - EvaluationsDynamic.__dimp2_gi(t, i, len(x))), 2) \
                - 2 * \
                math.cos(
                    3*math.pi*(x_i - EvaluationsDynamic.__dimp2_gi(t, i, len(x))))
        return 1 + 2 * (len(x) - 1) + sum

    @staticmethod
    def __dimp2_gi(t, i, n):
        return_value = math.pow(
            (math.sin(0.5 * math.pi * t + 2 * math.pi * (i / (n + 1)))), 2)
        return return_value

    @staticmethod
    def __dimp2_h(x, t):
        return_value = 1 - \
            math.sqrt(EvaluationsDynamic.__dimp2_f1(x, t) /
                      EvaluationsDynamic.__dimp2_g(x, t))
        return return_value

    @staticmethod
    def __dimp2_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__dimp2_g(
            x, t) * EvaluationsDynamic.__dimp2_h(x, t)
        return return_value

    def dimp2(self):
        # set loggin informatin
        self.__current_bench = self.__current_bench = 'bench_0_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        wandb.config.bench = 0
        # number of dimensions
        self.__num_dimensions = 2
        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dimp2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dimp2_f2)
        # objective name 1
        self.__objective_names.append('bench_0_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_0_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__dimp2_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for _ in range(1, self.__num_dimensions):
            self.__bounds.append([-2, 2])

    def dimp2_generate_pof(self, max_samples):
      """
      Return an array of pof's. 
          - Each pof has a list for f1 and a list for f2
      """
      pof = []
      f1 = []
      f2 = []
      for sample_index in range(max_samples):
        new_f1 = 1 / 1000 * sample_index
        new_f2 = (1 - math.sqrt(new_f1))
        f2.append(new_f2)
        f1.append(new_f1)
      pof.append([[f1, f2]])
      return
        

    def dimp2_generate_pos(self, max_samples):
      """
      Return pos'. 
          - [list for each g and a list for f1]
      """
      self.__severity_of_change = 10
      self.__frequency_of_change = 10
      pos = []
      g = []
      f1 = []
      self.update_t(1)
      prev_t = self.get_t()
      for sample_index in range(1, max_samples, 1):
        self.update_t(sample_index)
        new_f1 = 1 / 1000 * sample_index
        f1.append(new_f1)
        new_t = self.get_t()
        if(new_t != prev_t):
            g.append(self.__dimp2_gi())
            prev_g = new_g

    # ----------------------------------------------- FDA1 ------------------------------------------------------------ #

    @staticmethod
    def __fda1_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda1_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __fda1_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow(x[x_i] - EvaluationsDynamic.__fda1_G(t), 2)
        return 1 + sum

    @staticmethod
    def __fda1_h(x, t):
        return_value = 1 - \
            math.sqrt(EvaluationsDynamic.__fda1_f1(x) /
                      EvaluationsDynamic.__fda1_g(x, t))
        return return_value

    @staticmethod
    def __fda1_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __fda1_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__fda1_g(
            x, t) * EvaluationsDynamic.__fda1_h(x, t)
        return return_value

    def fda1(self):
        self.__current_bench = self.__current_bench = 'bench_1_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        wandb.config.bench = 1
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda1_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda1_f2)
        # objective name 1
        self.__objective_names.append('bench_1_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_1_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__fda1_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def fda1_generate_pof(self, max_samples):
      """
      Return an array of pof's. 
          - Each pof has a list for f1 and a list for f2
      """
      pof = []
      f1 = []
      f2 = []
      for sample_index in range(max_samples):
        new_f1 = 1 / 1000 * sample_index
        new_f2 = (1 - math.sqrt(new_f1))
        f2.append(new_f2)
        f1.append(new_f1)
      pof.append({"f1":f1, "f2": f2})
      return pof
        
    def fda1_generate_pos(self, max_samples):
      """
      Return an array of pos'. 
          - Each pos has multiple lists for g and a list for f1
      """
      self.__severity_of_change = 10
      self.__frequency_of_change = 10
      pos = []
      g = []
      f1 = []
      self.update_t(1)
      prev_t = self.get_t()
      print(prev_t)
      g.append(self.__fda1_G(prev_t))
      for sample_index in range(1, max_samples, 1):
        self.update_t(sample_index)
        new_f1 = 1 / 1000 * sample_index
        f1.append(new_f1)
        new_t = self.get_t()
        if(new_t != prev_t):
          prev_t = new_t
          g.append(self.__fda1_G(prev_t))

      for g_i in g:
        pos.append({"f1": f1, "g":[g_i]*len(f1)})
      return pos
        

    # ----------------------------------------------- FDA1_zhou (ZJZ) ------------------------------------------------------------ #

    @staticmethod
    def __fda1_zhou_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda1_zhou_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __fda1_zhou_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow(
                (x[x_i] - EvaluationsDynamic.__fda1_zhou_G(t) - math.pow(
                    math.fabs(x[1]), EvaluationsDynamic.__fda1_zhou_H(t))), 2)
        return 1 + sum

    @staticmethod
    def __fda1_zhou_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __fda1_zhou_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__fda1_zhou_f1(x) /
                                     EvaluationsDynamic.__fda1_zhou_g(x, t)),
                                    EvaluationsDynamic.__fda1_zhou_H(t))
        return return_value

    @staticmethod
    def __fda1_zhou_H(t):
        return_value = 1.5 + EvaluationsDynamic.__fda1_zhou_G(t)
        return return_value

    @staticmethod
    def __fda1_zhou_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__fda1_zhou_g(
            x, t) * EvaluationsDynamic.__fda1_zhou_h(x, t)
        return return_value

    def fda1_zhou(self):
        self.__current_bench = self.__current_bench = 'bench_2_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda1_zhou_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda1_zhou_f2)
        # objective name 1
        self.__objective_names.append('bench_2_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_2_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__fda1_zhou_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def fda1_zhou_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    f_1 = 1 / 1000 * sample_index
                    H = EvaluationsDynamic.__fda1_zhou_H(self.get_t())
                    pof.append(1 - math.pow(f_1, H))
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- FDA2 ------------------------------------------------------------ #

    @staticmethod
    def __fda2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda2_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __fda2_g(x):
        sum = 0
        for x_i in range(1, 2):
            sum += math.pow((x[x_i]), 2)
        return 1 + sum

    @staticmethod
    def __fda2_h(x, t):
        return_value = 1 - math.pow(
            (EvaluationsDynamic.__fda2_f1(x) / EvaluationsDynamic.__fda2_g(x)),
            EvaluationsDynamic.__fda2_H_2(x, t))
        return return_value

    @staticmethod
    def __fda2_H(t):
        return_value = 0.75 + 0.75 * math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __fda2_H_2(x, t):
        sum = 0
        for x_i in range(2, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__fda2_H(t)), 2)
        H = EvaluationsDynamic.__fda2_H(t)
        if (H == 0 and sum == 0):
            return 0
        return_value = math.pow((H + sum), -1)
        return return_value

    @staticmethod
    def __fda2_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__fda2_g(
            x) * EvaluationsDynamic.__fda2_h(x, t)
        return return_value

    def fda2(self):
        self.__current_bench = self.__current_bench = 'bench_3_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 3

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda2_f2)
        # objective name 1
        self.__objective_names.append('bench_3_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_3_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__fda2_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def fda2_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    H = EvaluationsDynamic.__fda2_H(self.get_t())
                    x_vector = [x_1, 0, H]
                    f_2 = EvaluationsDynamic.__fda2_f2(x_vector, self.get_t())

                    pof.append(f_2)
                if max(pof) != 0:
                    self.__save_true_pof(pof, pof_count)
                    pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- FDA2_camara ------------------------------------------------------------ #

    @staticmethod
    def __fda2_camara_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda2_camara_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __fda2_camara_g(x):
        sum = 0
        for x_i in range(1, 2):
            sum += math.pow((x[x_i]), 2)
        return 1 + sum

    @staticmethod
    def __fda2_camara_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__fda2_camara_f1(x) /
                                     EvaluationsDynamic.__fda2_camara_g(x)),
                                    EvaluationsDynamic.__fda2_camara_H_2(x, t))
        return return_value

    @staticmethod
    def __fda2_camara_H(t):
        return_value = math.pow(5, -1 * (math.cos(math.pi * t / 4)))
        return return_value

    @staticmethod
    def __fda2_camara_H_2(x, t):
        sum = 0
        for x_i in range(2, len(x)):
            sum += math.pow(
                (x[x_i] - (EvaluationsDynamic.__fda2_camara_H(t) / 2)), 2)
        return_value = EvaluationsDynamic.__fda2_camara_H(t) + sum
        return return_value

    @staticmethod
    def __fda2_camara_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__fda2_camara_g(
            x) * EvaluationsDynamic.__fda2_camara_h(x, t)
        return return_value

    def fda2_camara(self):
        self.__current_bench = 'bench_4_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 3

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda2_camara_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda2_camara_f2)
        # objective name 1
        self.__objective_names.append('bench_4_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_4_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__fda2_camara_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def fda2_camara_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    f_1 = 1 / 1000 * sample_index
                    H = EvaluationsDynamic.__fda2_camara_H(self.get_t())
                    if H != 0:
                        pof.append(1 - math.pow(f_1, H))
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- FDA3 ------------------------------------------------------------ #

    @staticmethod
    def __fda3_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda3_f1(x, t, r_i=None):
        sum = 0
        for index in range(1):
            sum += math.pow((x[index]), EvaluationsDynamic.__fda3_F(t))
        return sum

    @staticmethod
    def __fda3_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__fda3_G(t)), 2)
        return 1 + EvaluationsDynamic.__fda3_G(t) + sum

    @staticmethod
    def __fda3_h(x, t):
        return_value = 1 - \
            math.sqrt(EvaluationsDynamic.__fda3_f1(x, t) /
                      EvaluationsDynamic.__fda3_g(x, t))
        return return_value

    @staticmethod
    def __fda3_G(t):
        return_value = math.fabs(math.sin(0.5 * math.pi * t))
        return return_value

    @staticmethod
    def __fda3_F(t):
        return_value = math.pow(10, (2 * math.sin(0.5 * math.pi * t)))
        return return_value

    @staticmethod
    def __fda3_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__fda3_g(
            x, t) * EvaluationsDynamic.__fda3_h(x, t)
        return return_value

    def fda3(self):
        self.__current_bench = 'bench_5_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda3_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda3_f2)
        # objective name 1
        self.__objective_names.append('bench_5_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_5_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__fda3_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def fda3_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    f_1 = 1 / 1000 * sample_index
                    g = EvaluationsDynamic.__fda3_G(self.get_t())
                    pof.append((1 + g) * (1 - math.sqrt(f_1 / (1 + g))))
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- FDA3_camara ------------------------------------------------------------ #

    @staticmethod
    def __fda3_camara_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda3_camara_f1(x, t, r_i=None):
        return math.pow(x[0], EvaluationsDynamic.__fda3_camara_F(t))

    @staticmethod
    def __fda3_camara_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__fda3_camara_G(t)),
                            2)
        return 1 + EvaluationsDynamic.__fda3_camara_G(t) + sum

    @staticmethod
    def __fda3_camara_h(x, t):
        return_value = 1 - \
            math.sqrt(EvaluationsDynamic.__fda3_camara_f1(x, t) /
                      EvaluationsDynamic.__fda3_camara_g(x, t))
        return return_value

    @staticmethod
    def __fda3_camara_G(t):
        return_value = math.fabs(math.sin(0.5 * math.pi * t))
        return return_value

    @staticmethod
    def __fda3_camara_F(t):
        return_value = math.pow(10, 2 * math.sin(0.5 * math.pi * t))
        return return_value

    @staticmethod
    def __fda3_camara_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__fda3_camara_g(
            x, t) * EvaluationsDynamic.__fda3_camara_h(x, t)
        return return_value

    def fda3_camara(self):
        self.__current_bench = 'bench_6_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda3_camara_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda3_camara_f2)
        # objective name 1
        self.__objective_names.append('bench_6_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_6_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__fda3_camara_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def fda3_camara_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    f_1 = 1 / 1000 * sample_index
                    g = EvaluationsDynamic.__fda3_G(self.get_t())
                    pof.append((1 + g) * (1 - math.sqrt(f_1 / (1 + g))))
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- DMOP2 ------------------------------------------------------------ #
    @staticmethod
    def __dmop2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop2_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __dmop2_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__dmop2_G(t)), 2)
        return 1 + (9 * sum)

    @staticmethod
    def __dmop2_h(x, t):
        res_1 = EvaluationsDynamic.__dmop2_f1(
            x) / EvaluationsDynamic.__dmop2_g(x, t)
        return_value = 1 - math.pow(res_1, EvaluationsDynamic.__dmop2_H(t))
        return return_value

    @staticmethod
    def __dmop2_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __dmop2_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop2_f2(x, t, r_i=None):
        return_value = EvaluationsDynamic.__dmop2_g(
            x, t) * EvaluationsDynamic.__dmop2_h(x, t)
        return return_value

    def dmop2(self):
        self.__current_bench = 'bench_7_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop2_f2)
        # objective name 1
        self.__objective_names.append('bench_7_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_7_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__dmop2_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def dmop2_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    G = EvaluationsDynamic.__dmop2_dec_G(self.get_t())
                    x_vector = [x_1, G]
                    f_2 = EvaluationsDynamic.__dmop2_f2(x_vector, self.get_t())
                    # check if f_2 is bigger than any other f_2 in the pof
                    pof.append(f_2)
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- DMOP3 ------------------------------------------------------------ #
    @staticmethod
    def __dmop3_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop3_f1(x, t=None, r_i=None):
        return x[r_i]

    @staticmethod
    def __dmop3_g(x, t, r_i):
        sum = 0
        for x_i in range(0, len(x)):
            if x_i != r_i:
                sum += math.pow((x[x_i] - EvaluationsDynamic.__dmop3_G(t)), 2)
        return 1 + 9 * sum

    @staticmethod
    def __dmop3_h(x, t, r_i):
        return_value = 1 - math.sqrt(
            EvaluationsDynamic.__dmop3_f1(x, None, r_i) /
            EvaluationsDynamic.__dmop3_g(x, t, r_i))
        return return_value

    @staticmethod
    def __dmop3_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop3_f2(x, t, r_i):
        return_value = EvaluationsDynamic.__dmop3_g(
            x, t, r_i) * EvaluationsDynamic.__dmop3_h(x, t, r_i)
        return return_value

    def dmop3(self):
        self.__current_bench = 'bench_8_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 1
        self.__r_i = random.randint(0, self.__num_dimensions)

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop3_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop3_f2)
        # objective name 1
        self.__objective_names.append('bench_8_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_8_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__dmop3_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def dmop3_generate_pof(self, max_samples):
        pof = []
        for sample_index in range(max_samples):
            pof.append(1 - math.sqrt(1 / 1000 * sample_index))

        self.__save_true_pof(pof, 0)

    # ----------------------------------------------- DMOP2_iso ------------------------------------------------------------ #
    @staticmethod
    def __dmop2_iso_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop2_iso_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __dmop2_iso_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((EvaluationsDynamic.__dmop2_iso_yi(x[x_i], t) -
                             EvaluationsDynamic.__dmop2_iso_G(t)), 2)
        return 1 + 9 * sum

    @staticmethod
    def __dmop2_iso_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__dmop2_iso_f1(x) /
                                     EvaluationsDynamic.__dmop2_iso_g(x, t)),
                                    EvaluationsDynamic.__dmop2_iso_H(t))
        return return_value

    @staticmethod
    def __dmop2_iso_yi(x_i, t):
        A = EvaluationsDynamic.__dmop2_iso_G(t)
        B = 0.001
        C = 0.05
        y = (1 - EvaluationsDynamic.__dmop2_iso_H(t))
        return_value = A + min(0, math.floor(x_i-B))*((A*(B - x_i))/B) - \
            min(0, math.floor(C - y))*(((1 - A)*(x_i - C))/(1 - C))
        return return_value

    @staticmethod
    def __dmop2_iso_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __dmop2_iso_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop2_iso_f2(x, t=None, r_i=None):
        return_value = EvaluationsDynamic.__dmop2_iso_g(
            x, t) * EvaluationsDynamic.__dmop2_iso_h(x, t)
        return return_value

    def dmop2_iso(self):
        self.__current_bench = 'bench_9_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop2_iso_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop2_iso_f2)
        # objective name 1
        self.__objective_names.append('bench_9_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_9_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__dmop2_iso_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def dmop2_iso_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    f_1 = 1 / 1000 * sample_index
                    H = EvaluationsDynamic.__dmop2_H(self.get_t())
                    pof.append(1 - math.pow(f_1, H))
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- DMOP2_dec ------------------------------------------------------------ #
    @staticmethod
    def __dmop2_dec_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop2_dec_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __dmop2_dec_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((EvaluationsDynamic.__dmop2_dec_yi(t) -
                             EvaluationsDynamic.__dmop2_dec_G(t)), 2)
        return 1 + 9 * sum

    @staticmethod
    def __dmop2_dec_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__dmop2_dec_f1(x) /
                                     EvaluationsDynamic.__dmop2_dec_g(x, t)),
                                    EvaluationsDynamic.__dmop2_dec_H(t))
        return return_value

    @staticmethod
    def __dmop2_dec_yi(t):
        H = EvaluationsDynamic.__dmop2_dec_H(t)
        y = 1 - H
        a = 0.35
        b = 0.001
        c = 0.05
        return_value = (((math.floor(y - a + b) * (1 - c + ((a - b) / b))) /
                         (a - b)) + (1 / b) +
                        ((math.floor(a + b - y) * (1 - c +
                                                   ((1 - a - b) / b))) /
                         (1 - a - b))) * (math.fabs(y - a) - b) + 1
        return return_value

    @staticmethod
    def __dmop2_dec_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __dmop2_dec_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop2_dec_f2(x, t=None, r_i=None):
        return_value = EvaluationsDynamic.__dmop2_dec_g(
            x, t) * EvaluationsDynamic.__dmop2_dec_h(x, t)
        return return_value

    def dmop2_dec(self):
        self.__current_bench = 'bench_10_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop2_dec_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop2_dec_f2)
        # objective name 1
        self.__objective_names.append('bench_10_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_10_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__dmop2_dec_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def dmop2_dec_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    G = EvaluationsDynamic.__dmop2_dec_G(self.get_t())
                    x_vector = [x_1, G]
                    f_2 = EvaluationsDynamic.__dmop2_f2(x_vector, self.get_t())
                    # check if f_2 is bigger than any other f_2 in the pof
                    pof.append(f_2)
                self.__save_true_pof(pof, pof_count)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- HE1 ------------------------------------------------------------ #
    @staticmethod
    def __he_1_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_1_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __he_1_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += x[x_i]
        return 1 + (9 / (len(x))) * sum

    @staticmethod
    def __he_1_h(x):
        return_value = 1 - math.sqrt(
            EvaluationsDynamic.__he_1_f1(x) / EvaluationsDynamic.__he_1_g(x)
        ) - (EvaluationsDynamic.__he_1_f1(x) / EvaluationsDynamic.__he_1_g(x)
             ) * math.sin(10 * math.pi * EvaluationsDynamic.__he_1_f1(x))
        return return_value

    @staticmethod
    def __he_1_f2(x, t=None, r_i=None):
        return_value = EvaluationsDynamic.__he_1_g(
            x) * EvaluationsDynamic.__he_1_h(x)
        return return_value

    def he_1(self):
        self.__current_bench = 'bench_11_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_1_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_1_f2)
        # objective name 1
        self.__objective_names.append('bench_11_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_11_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__he_1_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def he_1_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                pof_x = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    x_vector = [x_1, 0]
                    f_2 = EvaluationsDynamic.__he_1_f2(x_vector, self.get_t())
                    if pof:
                        if all(i > f_2 for i in pof):
                            pof.append(f_2)
                            pof_x.append(x_1)
                        else:
                            test = 1
                    else:
                        pof.append(f_2)
                        pof_x.append(x_1)
                self.__save_true_pof(pof, pof_count, pof_x)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- HE2 ------------------------------------------------------------ #
    @staticmethod
    def __he_2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_2_f1(x, t=None, r_i=None):
        return x[0]

    @staticmethod
    def __he_2_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += x[x_i]
        return 1 + (9 / (len(x))) * sum

    @staticmethod
    def __he_2_h(x, t):
        return_value = 1 - math.pow(
            math.sqrt(
                EvaluationsDynamic.__he_2_f1(x) /
                EvaluationsDynamic.__he_2_g(x)), EvaluationsDynamic.__h_2_H(t)
        ) - math.pow(
            (EvaluationsDynamic.__he_2_f1(x) / EvaluationsDynamic.__he_2_g(x)),
            EvaluationsDynamic.__h_2_H(t)) * math.sin(
                10 * math.pi * EvaluationsDynamic.__he_2_f1(x))
        return return_value

    @staticmethod
    def __h_2_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __he_2_f2(x, t=None, r_i=None):
        return_value = EvaluationsDynamic.__he_2_g(
            x) * EvaluationsDynamic.__he_2_h(x, t)
        return return_value

    def he_2(self):
        self.__current_bench = 'bench_12_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 2

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_2_f2)
        # objective name 1
        self.__objective_names.append('bench_12_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_12_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__he_2_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def he_2_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                pof_x = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    x_vector = [x_1, 0]
                    f_2 = EvaluationsDynamic.__he_2_f2(x_vector, self.get_t())
                    # check if f_2 is bigger than any other f_2 in the pof
                    if pof:
                        if all(i > f_2 for i in pof):
                            pof.append(f_2)
                            pof_x.append(x_1)
                    else:
                        pof.append(f_2)
                        pof_x.append(x_1)
                self.__save_true_pof(pof, pof_count, pof_x)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- HE3 ------------------------------------------------------------ #
    @staticmethod
    def __he_3_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_3_f1(x, t=None, r_i=None):
        sum_j = 0
        j_size = 0
        for x_i in range(1, len(x)):
            if (x_i + 1) % 2 != 0:
                sum_j += math.pow(
                    (x[x_i] - math.pow(x[0],
                                       (0.5 * (1 + ((3 * ((x_i + 1) - 2)) /
                                                    (len(x) - 2)))))), 2)
                j_size += 1
        return x[0] + (2 / j_size) * sum_j

    @staticmethod
    def __he_3_g(x):
        sum_j = 0
        j_size = 0
        for x_i in range(1, len(x)):
            if (x_i + 1) % 2 == 0:
                sum_j += math.pow(
                    (x[x_i] - math.pow(x[0],
                                       (0.5 * (1 + ((3 * ((x_i + 1) - 2)) /
                                                    (len(x) - 2)))))), 2)
                j_size += 1
        return 2 - math.sqrt(x[0]) + (2 / j_size) * sum_j

    @staticmethod
    def __he_3_h(x, t):
        return_value = 1 - math.pow(
            (EvaluationsDynamic.__he_3_f1(x) / EvaluationsDynamic.__he_3_g(x)),
            EvaluationsDynamic.__he_3_H(t))
        return return_value

    @staticmethod
    def __he_3_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __he_3_f2(x, t=None, r_i=None):
        return_value = EvaluationsDynamic.__he_3_g(
            x) * EvaluationsDynamic.__he_3_h(x, t)
        return return_value

    def he_3(self):
        self.__current_bench = 'bench_13_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 3

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_3_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_3_f2)
        # objective name 1
        self.__objective_names.append('bench_13_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_13_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__he_3_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = []
        for loop in range(self.__num_dimensions):
            self.__bounds.append([0, 1])

    def he_3_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                pof_x = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    x_vector = [x_1]
                    for x_i in range(1, self.__num_dimensions):
                        pos = math.pow(x_1,
                                       (0.5 * (3 * ((x_i + 1) - 2) /
                                               (self.__num_dimensions - 2))))
                        x_vector.append(pos)
                    f_2 = EvaluationsDynamic.__he_6_f2(x_vector, self.get_t())
                    # check if f_2 is bigger than any other f_2 in the pof
                    if pof:
                        if all(i > f_2 for i in pof):
                            pof.append(f_2)
                            pof_x.append(x_1)
                    else:
                        pof.append(f_2)
                        pof_x.append(x_1)
                self.__save_true_pof(pof, pof_count, pof_x)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- HE6 ------------------------------------------------------------ #
    @staticmethod
    def __he_6_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_6_f1(x, t=None, r_i=None):
        sum_j = 0
        j_size = 0
        for x_i in range(1, len(x)):
            if (x_i + 1) % 2 != 0:
                sum_j += math.pow((x[x_i] - (0.8 * x[0] * math.cos(
                    (6 * math.pi * x[0] +
                     ((x_i + 1) * math.pi / len(x))) / 3))), 2)
                j_size += 1
        return x[0] + (2 / j_size) * sum_j

    @staticmethod
    def __he_6_g(x):
        sum_j = 0
        j_size = 0
        for x_i in range(1, len(x)):
            if (x_i + 1) % 2 == 0:
                sum_j += math.pow(
                    (x[x_i] - 0.8 * math.cos(6 * math.pi * x[0] +
                                             ((x_i + 1) * math.pi / len(x)))),
                    2)
                j_size += 1
        return 2 - math.sqrt(x[0]) + (2 / j_size) * sum_j

    @staticmethod
    def __he_6_h(x, t):
        f_1 = EvaluationsDynamic.__he_6_f1(x)
        g = EvaluationsDynamic.__he_6_g(x)
        H = EvaluationsDynamic.__he_6_H(t)
        res = f_1 / g
        return_value = 1 - math.pow(res, H)
        return return_value

    @staticmethod
    def __he_6_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __he_6_f2(x, t=None, r_i=None):
        g = EvaluationsDynamic.__he_6_g(x)
        h = EvaluationsDynamic.__he_6_h(x, t)
        return_value = g * h
        return return_value

    def he_6(self):
        self.__current_bench = 'bench_14_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 3

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_6_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_6_f2)
        # objective name 1
        self.__objective_names.append('bench_14_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_14_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__he_6_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def he_6_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                pof_x = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    x_vector = [x_1]
                    for x_i in range(1, self.__num_dimensions):
                        if (x_i + 1) % 2 != 0:
                            pos = 0.8 * x_1 * \
                                math.cos(
                                    ((6 * math.pi * x_1) + ((x_i+1)*math.pi / self.__num_dimensions)) / 3)
                        else:
                            pos = 0.8 * x_1 * \
                                math.sin((6 * math.pi * x_1) + ((x_i+1)
                                                                * math.pi / self.__num_dimensions))
                        x_vector.append(pos)
                    f_2 = EvaluationsDynamic.__he_6_f2(x_vector, self.get_t())
                    # check if f_2 is bigger than any other f_2 in the pof
                    if pof:
                        if all(i > f_2 for i in pof):
                            pof.append(f_2)
                            pof_x.append(x_1)
                    else:
                        pof.append(f_2)
                        pof_x.append(x_1)
                self.__save_true_pof(pof, pof_count, pof_x)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- HE7 ------------------------------------------------------------ #
    @staticmethod
    def __he_7_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_7_f1(x, t=None, r_i=None):
        sum_j = 0
        j = 0
        for x_i in range(1, len(x)):
            if (x_i + 1) % 2 != 0:
                sum_j += math.pow((
                    x[x_i] -
                    (0.3 * math.pow(x[0], 2) * math.cos(24 * math.pi * x[0] +
                                                        (4 *
                                                         (x_i + 1) / len(x))) +
                     0.6 * x[0]) * math.cos(6 * math.pi * x[0] +
                                            ((x_i + 1) * math.pi / len(x)))),
                    2)
                j += 1
        return x[0] + (2 / j) * sum_j

    @staticmethod
    def __he_7_g(x):
        sum_j = 0
        j = 0
        for x_i in range(1, len(x)):
            if (x_i + 1) % 2 == 0:
                sum_j += math.pow((
                    x[x_i] -
                    (0.3 * math.pow(x[0], 2) * math.cos(24 * math.pi * x[0] +
                                                        (4 *
                                                         (x_i + 1) / len(x))) +
                     0.6 * x[0]) * math.sin(6 * math.pi * x[0] +
                                            ((x_i + 1) * math.pi / len(x)))),
                    2)
                j += 1
        return 2 - math.sqrt(x[0]) + (2 / j) * sum_j

    @staticmethod
    def __he_7_h(x, t):
        return_value = 1 - math.pow(
            (EvaluationsDynamic.__he_7_f1(x) / EvaluationsDynamic.__he_7_g(x)),
            EvaluationsDynamic.__he_7_H(t))
        return return_value

    @staticmethod
    def __he_7_H(t):
        return_value = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        return return_value

    @staticmethod
    def __he_7_f2(x, t=None, r_i=None):
        return_value = EvaluationsDynamic.__he_7_g(
            x) * EvaluationsDynamic.__he_7_h(x, t)
        return return_value

    def he_7(self):
        self.__current_bench = 'bench_15_nt_' + \
            str(self.__severity_of_change) + '_Tt_' + \
            str(self.__frequency_of_change)
        # number of dimensions
        self.__num_dimensions = 3

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_7_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_7_f2)
        # objective name 1
        self.__objective_names.append('bench_15_obj_1_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_15_obj_2_nt_' +
                                      str(self.__severity_of_change) + '_Tt_' +
                                      str(self.__frequency_of_change))
        # objective type 1
        self.__objective_types.append("min")
        # objective type 2
        self.__objective_types.append("min")
        # number of particles
        self.__num_particles = [25, 25]
        # constants
        self.__constants = EvaluationsDynamic.__he_7_constants()
        # selection size
        self.__tournament_selection_size = 3
        # bounds
        self.__bounds = [[0, 1]]
        for loop in range(self.__num_dimensions):
            self.__bounds.append([-1, 1])

    def he_7_generate_pof(self, max_samples):
        pof_count = 0
        prev_t = -1
        for iteration in range(1000):
            print(iteration)
            self.update_t(iteration + 1)
            if self.get_t() != prev_t:
                pof = []
                pof_x = []
                for sample_index in range(max_samples):
                    x_1 = 1 / 1000 * sample_index
                    x_vector = [x_1]
                    for x_i in range(1, self.__num_dimensions):
                        a = 0.3 * math.pow(x_1, 2) * math.cos(
                            24 * math.pi * x_1 +
                            (4 *
                             (x_i + 1) / self.__num_dimensions)) + 0.6 * x_1
                        if (x_i + 1) % 2 != 0:
                            pos = a * \
                                math.cos(
                                    ((6 * math.pi * x_1) + ((x_i + 1) * math.pi / self.__num_dimensions)) / 3)
                        else:
                            pos = a * \
                                math.cos((6 * math.pi * x_1) + ((x_i + 1)
                                                                * math.pi / self.__num_dimensions))
                        x_vector.append(pos)
                    f_2 = EvaluationsDynamic.__he_7_f2(x_vector, self.get_t())
                    # check if f_2 is bigger than any other f_2 in the pof
                    if pof:
                        if all(i > f_2 for i in pof):
                            pof.append(f_2)
                            pof_x.append(x_1)
                    else:
                        pof.append(f_2)
                        pof_x.append(x_1)
                self.__save_true_pof(pof, pof_count, pof_x)
                pof_count += 1
                prev_t = self.get_t()

    # ----------------------------------------------- Extra ------------------------------------------------------------ #
    def __save_true_pof(self, pof, iteration, pof_x=None):
        file_writer = open(
            "TRUE DYNAMIC POF/" + self.__current_bench + "_pof_" +
            str(iteration), 'w')
        if pof_x:
            file_writer_2 = open(
                "TRUE DYNAMIC POF/" + self.__current_bench + "_pof_x_" +
                str(iteration), 'w')
            file_writer_2.close()
        file_writer.close()
        for index in range(len(pof)):
            file_writer = open(
                "TRUE DYNAMIC POF/" + self.__current_bench + "_pof_" +
                str(iteration), 'a')
            file_writer.write("%s\n" % pof[index])
            file_writer.close()
            if pof_x:
                file_writer_2 = open(
                    "TRUE DYNAMIC POF/" + self.__current_bench + "_pof_x_" +
                    str(iteration), 'a')
                file_writer_2.write("%s\n" % pof_x[index])
                file_writer_2.close()
        return
