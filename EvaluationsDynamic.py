import math

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

    def set_frequency_of_change(self, f):
        self.__frequency_of_change = f

    def update_t(self, iteration):
        self.__t = (1/self.__severity_of_change)*math.floor(iteration/self.__frequency_of_change)
        return

    def get_t(self):
        return self.__t

    def set_run(self, run):
        self.__run_number = run
        return

    def get_run(self):
        return self.__run_number

    # ----------------------------------------------- DIMP2 ------------------------------------------------------------ #
    @staticmethod
    def __dimp2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dimp2_f1(x, t = None):
        return x[0]

    @staticmethod
    def __dimp2_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__dimp2_gi(t, x_i+1, len(x))), 2) \
                   - 2*math.cos(3*math.pi*(x[x_i] - EvaluationsDynamic.__dimp2_gi(t, x_i+1, len(x))))
        return 1 + 2*(len(x)-1) + sum

    @staticmethod
    def __dimp2_gi(t, i, n):
        return_value = math.pow((math.sin(0.5*math.pi*t + 2*math.pi*(i/n+1))), 2)
        return return_value

    @staticmethod
    def __dimp2_h(x, t):
        return_value = 1 - math.sqrt(EvaluationsDynamic.__dimp2_f1(x, t) / EvaluationsDynamic.__dimp2_g(x, t))
        return return_value

    @staticmethod
    def __dimp2_f2(x, t):
        return_value = EvaluationsDynamic.__dimp2_g(x, t) * EvaluationsDynamic.__dimp2_h(x, t)
        return return_value

    def dimp2(self):
        self.__current_bench = "dimp2"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dimp2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dimp2_f2)
        # objective name 1
        self.__objective_names.append('bench_0_obj_1_nt_'+str(self.__severity_of_change)+'_Tt_'+str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_0_obj_2_nt_'+str(self.__severity_of_change)+'_Tt_'+str(self.__frequency_of_change))
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
        for loop in range(1, self.__num_dimensions):
            self.__bounds.append([-2, 2])

    # ----------------------------------------------- FDA1_zhou ------------------------------------------------------------ #

    @staticmethod
    def __fda1_zhou_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda1_zhou_f1(x):
        return x[0]

    @staticmethod
    def __fda1_zhou_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__dimp2_gi(t, x_i+1, len(x)) -
                             math.pow(x[0], EvaluationsDynamic.__fda1_zhou_H(t))), 2)
        return 1 + sum

    @staticmethod
    def __fda1_zhou_G(t):
        return_value = math.sin(0.5*math.pi*t)
        return return_value

    @staticmethod
    def __fda1_zhou_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__dimp2_f1(x)/EvaluationsDynamic.__dimp2_g(x, t)), EvaluationsDynamic.__fda1_zhou_H(t))
        return return_value

    @staticmethod
    def __fda1_zhou_H(t):
        return_value = 1.5 + EvaluationsDynamic.__fda1_zhou_G(t)
        return return_value

    @staticmethod
    def __fda1_zhou_f2(x, t):
        return_value = EvaluationsDynamic.__fda1_zhou_g(x, t) * EvaluationsDynamic.__fda1_zhou_h(x, t)
        return return_value

    def fda1_zhou(self):
        self.__current_bench = "FDA1_zhou"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda1_zhou_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda1_zhou_f2)
        # objective name 1
        self.__objective_names.append('bench_1_obj_1_nt_' + str(self.__severity_of_change) + '_Tt_' + str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_1_obj_2_nt_' + str(self.__severity_of_change) + '_Tt_' + str(self.__frequency_of_change))
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

    # ----------------------------------------------- FDA2 ------------------------------------------------------------ #

    @staticmethod
    def __fda2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda2_f1(x, t=None):
        return x[0]

    @staticmethod
    def __fda2_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i]), 2)
        return 1 + sum

    @staticmethod
    def __fda2_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__fda2_f1(x)/EvaluationsDynamic.__fda2_g(x)), EvaluationsDynamic.__fda2_H_2(x, t))
        return return_value

    @staticmethod
    def __fda2_H(t):
        return_value = 0.75 + 0.75*math.sin(0.5*math.pi*t)
        return return_value

    @staticmethod
    def __fda2_H_2(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__fda2_H(t)), 2)
        return_value = math.pow((EvaluationsDynamic.__fda2_H(t) + sum), -1)
        return return_value

    @staticmethod
    def __fda2_f2(x, t):
        return_value = EvaluationsDynamic.__fda2_g(x) * EvaluationsDynamic.__fda2_h(x, t)
        return return_value

    def fda2(self):
        self.__current_bench = "FDA2"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda2_f2)
        # objective name 1
        self.__objective_names.append('bench_2_obj_1_nt_' + str(self.__severity_of_change) + '_Tt_' + str(self.__frequency_of_change))
        # objective name 2
        self.__objective_names.append('bench_2_obj_2_nt_' + str(self.__severity_of_change) + '_Tt_' + str(self.__frequency_of_change))
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

    # ----------------------------------------------- FDA2_camara ------------------------------------------------------------ #

    @staticmethod
    def __fda2_camara_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda2_camara_f1(x):
        return x[0]

    @staticmethod
    def __fda2_camara_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i]), 2)
        return 1 + sum

    @staticmethod
    def __fda2_camara_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__fda2_camara_f1(x)/EvaluationsDynamic.__fda2_camara_g(x)), EvaluationsDynamic.__fda2_camara_H_2(x, t))
        return return_value

    @staticmethod
    def __fda2_camara_H(t):
        return_value = math.pow(5, -1*(math.cos(math.pi*t/4)))
        return return_value

    @staticmethod
    def __fda2_camara_H_2(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - (EvaluationsDynamic.__fda2_camara_H(t)/2)), 2)
        return_value = EvaluationsDynamic.__fda2_camara_H(t) + sum
        return return_value

    @staticmethod
    def __fda2_camara_f2(x, t):
        return_value = EvaluationsDynamic.__fda2_camara_g(x) * EvaluationsDynamic.__fda2_camara_h(x, t)
        return return_value

    def fda2_camara(self):
        self.__current_bench = "FDA2_camara"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda2_camara_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda2_camara_f2)
        # objective name 1
        self.__objective_names.append('FDA2_camara_1')
        # objective name 2
        self.__objective_names.append('FDA2_camara_2')
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

    # ----------------------------------------------- FDA3 ------------------------------------------------------------ #

    @staticmethod
    def __fda3_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda3_f1(x, t):
        sum = 0
        for x_i in range(len(x)):
            sum += math.pow((x[x_i]), EvaluationsDynamic.__fda3_F(t))
        return sum

    @staticmethod
    def __fda3_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__fda3_G(t)), 2)
        return 1 + EvaluationsDynamic.__fda3_G(t) + sum

    @staticmethod
    def __fda3_h(x, t):
        return_value = 1 - math.sqrt(EvaluationsDynamic.__fda3_f1(x, t)/EvaluationsDynamic.__fda3_g(x, t))
        return return_value

    @staticmethod
    def __fda3_G(t):
        return_value = math.fabs(math.sin(0.5*math.pi*t))
        return return_value

    @staticmethod
    def __fda3_F(t):
        return_value = math.pow(10, 2*math.sin(0.5*math.pi*t))
        return return_value

    @staticmethod
    def __fda3_f2(x, t):
        return_value = EvaluationsDynamic.__fda3_g(x, t) * EvaluationsDynamic.__fda3_h(x, t)
        return return_value

    def fda3(self):
        self.__current_bench = "FDA3"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda3_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda3_f2)
        # objective name 1
        self.__objective_names.append('FDA3_1')
        # objective name 2
        self.__objective_names.append('FDA3_2')
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

    # ----------------------------------------------- FDA3_camara ------------------------------------------------------------ #

    @staticmethod
    def __fda3_camara_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __fda3_camara_f1(x, t):
        return math.pow(x[0], EvaluationsDynamic.__fda3_camara_F(t))

    @staticmethod
    def __fda3_camara_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__fda3_camara_G(t)), 2)
        return 1 + EvaluationsDynamic.__fda3_camara_G(t) + sum

    @staticmethod
    def __fda3_camara_h(x, t):
        return_value = 1 - math.sqrt(EvaluationsDynamic.__fda3_camara_f1(x, t)/EvaluationsDynamic.__fda3_camara_g(x, t))
        return return_value

    @staticmethod
    def __fda3_camara_G(t):
        return_value = math.fabs(math.sin(0.5*math.pi*t))
        return return_value

    @staticmethod
    def __fda3_camara_F(t):
        return_value = math.pow(10, 2*math.sin(0.5*math.pi*t))
        return return_value

    @staticmethod
    def __fda3_camara_f2(x, t):
        return_value = EvaluationsDynamic.__fda3_camara_g(x, t) * EvaluationsDynamic.__fda3_camara_h(x, t)
        return return_value

    def fda3_camara(self):
        self.__current_bench = "FDA3_camara"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__fda3_camara_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__fda3_camara_f2)
        # objective name 1
        self.__objective_names.append('FDA3_camara_1')
        # objective name 2
        self.__objective_names.append('FDA3_camara_2')
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

    # ----------------------------------------------- DMOP2 ------------------------------------------------------------ #
    @staticmethod
    def __dmop2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop2_f1(x):
        return x[0]

    @staticmethod
    def __dmop2_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((x[x_i] - EvaluationsDynamic.__dmop2_G(t)), 2)
        return 1 + 9*sum

    @staticmethod
    def __dmop2_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__dmop2_f1(x) / EvaluationsDynamic.__dmop2_g(x, t)), EvaluationsDynamic.__dmop2_H(t))
        return return_value

    @staticmethod
    def __dmop2_H(t):
        return_value = 0.75*math.sin(0.5*math.pi*t) + 1.25
        return return_value

    @staticmethod
    def __dmop2_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop2_f2(x, t):
        return_value = EvaluationsDynamic.__dmop2_g(x, t) * EvaluationsDynamic.__dmop2_h(x, t)
        return return_value

    def dmop2(self):
        self.__current_bench = "dmop2"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop2_f2)
        # objective name 1
        self.__objective_names.append('dmop2_1')
        # objective name 2
        self.__objective_names.append('dmop2_2')
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

    # ----------------------------------------------- DMOP3 ------------------------------------------------------------ #
    @staticmethod
    def __dmop3_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop3_f1(x, r_i):
        return x[r_i]

    @staticmethod
    def __dmop3_g(x, t, r_i):
        sum = 0
        for x_i in range(0, len(x)):
            if x_i != r_i:
                sum += math.pow((x[x_i] - EvaluationsDynamic.__dmop3_G(t)), 2)
        return 1 + 9*sum

    @staticmethod
    def __dmop3_h(x, t, r_i):
        return_value = 1 - math.sqrt(EvaluationsDynamic.__dmop3_f1(x, r_i) / EvaluationsDynamic.__dmop3_g(x, t, r_i))
        return return_value

    @staticmethod
    def __dmop3_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop3_f2(x, t, r_i):
        return_value = EvaluationsDynamic.__dmop3_g(x, t, r_i) * EvaluationsDynamic.__dmop3_h(x, t, r_i)
        return return_value

    def dmop3(self):
        self.__current_bench = "dmop3"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop3_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop3_f2)
        # objective name 1
        self.__objective_names.append('dmop3_1')
        # objective name 2
        self.__objective_names.append('dmop3_2')
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

    # ----------------------------------------------- DMOP2_iso ------------------------------------------------------------ #
    @staticmethod
    def __dmop2_iso_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop2_iso_f1(x):
        return x[0]

    @staticmethod
    def __dmop2_iso_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((EvaluationsDynamic.__dmop2_iso_yi(x[x_i], t) - EvaluationsDynamic.__dmop2_iso_G(t)), 2)
        return 1 + 9*sum

    @staticmethod
    def __dmop2_iso_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__dmop2_iso_f1(x) / EvaluationsDynamic.__dmop2_iso_g(x, t)), EvaluationsDynamic.__dmop2_iso_H(t))
        return return_value

    @staticmethod
    def __dmop2_iso_yi(x_i, t):
        return_value = EvaluationsDynamic.__dmop2_iso_G(t) + min(0, math.floor(x_i-0.001))*((EvaluationsDynamic.__dmop2_iso_G(t)*(0.001 - x_i))/0.001) - \
                       min(0, math.floor(0.05 - (1-EvaluationsDynamic.__dmop2_iso_H(t))))*(((1 - EvaluationsDynamic.__dmop2_iso_G(t))*(x_i - 0.05))/(1 - 0.05))
        return return_value

    @staticmethod
    def __dmop2_iso_H(t):
        return_value = 0.75*math.sin(0.5*math.pi*t) + 1.25
        return return_value

    @staticmethod
    def __dmop2_iso_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop2_iso_f2(x, t):
        return_value = EvaluationsDynamic.__dmop2_iso_g(x, t) * EvaluationsDynamic.__dmop2_iso_h(x, t)
        return return_value

    def dmop2_iso(self):
        self.__current_bench = "dmop2_iso"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop2_iso_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop2_iso_f2)
        # objective name 1
        self.__objective_names.append('dmop2_iso_1')
        # objective name 2
        self.__objective_names.append('dmop2_iso_2')
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

    # ----------------------------------------------- DMOP2_dec ------------------------------------------------------------ #
    @staticmethod
    def __dmop2_dec_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __dmop2_dec_f1(x):
        return x[0]

    @staticmethod
    def __dmop2_dec_g(x, t):
        sum = 0
        for x_i in range(1, len(x)):
            sum += math.pow((EvaluationsDynamic.__dmop2_dec_yi(t) - EvaluationsDynamic.__dmop2_dec_G(t)), 2)
        return 1 + 9*sum

    @staticmethod
    def __dmop2_dec_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__dmop2_dec_f1(x) / EvaluationsDynamic.__dmop2_dec_g(x, t)), EvaluationsDynamic.__dmop2_dec_H(t))
        return return_value

    @staticmethod
    def __dmop2_dec_yi(t):
        y = 1-EvaluationsDynamic.__dmop2_dec_H(t)
        a = 0.35
        b = 0.001
        c = 0.05
        return_value = ((math.floor(y - a + b)*(1 - c + (a - b)/c))/(a - b) + (1/b) + (math.floor(a + b - y)*(1 - c + (1 - a - b)/b))/(1 - a - b))*(math.fabs(y - a) + 1)
        return return_value

    @staticmethod
    def __dmop2_dec_H(t):
        return_value = 0.75*math.sin(0.5*math.pi*t) + 1.25
        return return_value

    @staticmethod
    def __dmop2_dec_G(t):
        return_value = math.sin(0.5 * math.pi * t)
        return return_value

    @staticmethod
    def __dmop2_dec_f2(x, t):
        return_value = EvaluationsDynamic.__dmop2_dec_g(x, t) * EvaluationsDynamic.__dmop2_dec_h(x, t)
        return return_value

    def dmop2_dec(self):
        self.__current_bench = "dmop2_dec"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__dmop2_dec_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__dmop2_dec_f2)
        # objective name 1
        self.__objective_names.append('dmop2_dec_1')
        # objective name 2
        self.__objective_names.append('dmop2_dec_2')
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

    # ----------------------------------------------- HE1 ------------------------------------------------------------ #
    @staticmethod
    def __he_1_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_1_f1(x):
        return x[0]

    @staticmethod
    def __he_1_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += x[x_i]
        return 1 + (9/(len(x)-1))*sum

    @staticmethod
    def __he_1_h(x):
        return_value = 1 - math.sqrt(EvaluationsDynamic.__he_1_f1(x) / EvaluationsDynamic.__he_1_g(x)) - (EvaluationsDynamic.__he_1_f1(x) / EvaluationsDynamic.__he_1_g(x))*math.sin(10*math.pi*EvaluationsDynamic.__he_1_f1(x))
        return return_value

    @staticmethod
    def __he_1_f2(x):
        return_value = EvaluationsDynamic.__he_1_g(x) * EvaluationsDynamic.__he_1_h(x)
        return return_value

    def __he_1(self):
        self.__current_bench = "he_1"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_1_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_1_f2)
        # objective name 1
        self.__objective_names.append('he_1_1')
        # objective name 2
        self.__objective_names.append('he_1_2')
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

    # ----------------------------------------------- HE2 ------------------------------------------------------------ #
    @staticmethod
    def __he_2_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_2_f1(x):
        return x[0]

    @staticmethod
    def __he_2_g(x):
        sum = 0
        for x_i in range(1, len(x)):
            sum += x[x_i]
        return 1 + (9/(len(x)-1))*sum

    @staticmethod
    def __he_2_h(x, t):
        return_value = 1 - math.pow((math.sqrt(EvaluationsDynamic.__he_1_f1(x) /
            EvaluationsDynamic.__he_1_g(x))), EvaluationsDynamic.__h_2_H(t)) - math.pow((EvaluationsDynamic.__he_1_f1(x) /
            EvaluationsDynamic.__he_1_g(x)), EvaluationsDynamic.__h_2_H(t))*math.sin(10*math.pi*EvaluationsDynamic.__he_1_f1(x))
        return return_value

    @staticmethod
    def __h_2_H(t):
        return_value = 0.75*math.sin(0.5*math.pi*t) + 1.25
        return return_value

    @staticmethod
    def __he_2_f2(x):
        return_value = EvaluationsDynamic.__he_1_g(x) * EvaluationsDynamic.__he_1_h(x)
        return return_value

    def __he_2(self):
        self.__current_bench = "he_2"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_2_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_2_f2)
        # objective name 1
        self.__objective_names.append('he_2_1')
        # objective name 2
        self.__objective_names.append('he_2_2')
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

    # ----------------------------------------------- HE6 ------------------------------------------------------------ #
    @staticmethod
    def __he_6_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_6_f1(x):
        sum_j = 0
        j = 0
        for x_i in range(2, len(x)):
            if x_i % 2 != 0:
                sum_j += math.pow((x[x_i] - 0.8 * x[0] * math.cos((6 * math.pi * x[0] + (x_i * math.pi / len(x))) / 3)), 2)
                j += math.pow(x[x_i], 2)
        return x[0] + (2 / math.fabs(j))*sum_j

    @staticmethod
    def __he_6_g(x):
        sum_j = 0
        j = 0
        for x_i in range(2, len(x)):
            if x_i % 2 == 0:
                sum_j += math.pow((x[x_i] - 0.8*math.cos(6*math.pi*x[0]+(x_i*math.pi/len(x)))), 2)
                j += 1
        return 2 - math.sqrt(x[0]) + (2/j)*sum_j

    @staticmethod
    def __he_6_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__he_6_f1(x)/EvaluationsDynamic.__he_6_g(x)), EvaluationsDynamic.__he_6_H(t))
        return return_value

    @staticmethod
    def __he_6_H(t):
        return_value = 0.75*math.sin(0.5*math.pi*t) + 1.25
        return return_value

    @staticmethod
    def __he_6_f2(x):
        return_value = EvaluationsDynamic.__he_1_g(x) * EvaluationsDynamic.__he_1_h(x)
        return return_value

    def __he_6(self):
        self.__current_bench = "he_6"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_6_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_6_f2)
        # objective name 1
        self.__objective_names.append('he_6_1')
        # objective name 2
        self.__objective_names.append('he_6_2')
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

    # ----------------------------------------------- HE7 ------------------------------------------------------------ #
    @staticmethod
    def __he_7_constants():
        return [0.475, 1.80, 1.10, 1.80]

    @staticmethod
    def __he_7_f1(x):
        sum_j = 0
        j = 0
        for x_i in range(2, len(x)):
            if x_i % 2 != 0:
                sum_j += math.pow((x[x_i] - (0.3*math.pow(x[0], 2)*math.cos(24*math.pi*x[0] + (4*x_i/len(x))) + 0.6*x[0])*math.cos(6*math.pi*x[0] + (x_i*math.pi/len(x)))), 2)
                j += 1
        return x[0] + (2 / j)*sum_j

    @staticmethod
    def __he_7_g(x):
        sum_j = 0
        j = 0
        for x_i in range(2, len(x)):
            if x_i % 2 == 0:
                sum_j += math.pow((x[x_i] - (0.3 * math.pow(x[0], 2) * math.cos(24 * math.pi * x[0] + (4 * x_i / len(x))) + 0.6 * x[0]) * math.sin(6 * math.pi * x[0] + (x_i * math.pi / len(x)))), 2)
                j += j
        return 2 - math.sqrt(x[0]) + (2 / j) * sum_j

    @staticmethod
    def __he_7_h(x, t):
        return_value = 1 - math.pow((EvaluationsDynamic.__he_6_f1(x)/EvaluationsDynamic.__he_6_g(x)), EvaluationsDynamic.__he_6_H(t))
        return return_value

    @staticmethod
    def __he_7_H(t):
        return_value = 0.75*math.sin(0.5*math.pi*t) + 1.25
        return return_value

    @staticmethod
    def __he_7_f2(x):
        return_value = EvaluationsDynamic.__he_1_g(x) * EvaluationsDynamic.__he_1_h(x)
        return return_value

    def __he_7(self):
        self.__current_bench = "he_7"
        # number of dimensions
        self.__num_dimensions = 30

        # objective 1
        self.__objectives.append(EvaluationsDynamic.__he_7_f1)
        # objective 2
        self.__objectives.append(EvaluationsDynamic.__he_7_f2)
        # objective name 1
        self.__objective_names.append('he_7_1')
        # objective name 2
        self.__objective_names.append('he_7_2')
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

