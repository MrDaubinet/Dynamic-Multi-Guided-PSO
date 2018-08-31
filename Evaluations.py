from optproblems import zdt
from optproblems import wfg


class Evaluations:
    def __init__(self):
        self._objectives = []
        self._objective_types = []
        self._bounds = []
        self.num_particles = []
        self.num_dimensions = None
        self._objective_names = []
        self._constants = []
        self._tournament_selection_size = None
        self._extra_bounds = None
        self._current_bench = None
        self._true_pof = None

# ------------------------------------------------------------------------------------------------- Class Functions ----------------------------------------------------------------------------------------------------- #
    def get_objective_functions(self):
        return self._objectives

    def get_objective_names(self):
        return self._objective_names

    def num_objectives(self):
        return len(self._objectives)

    def get_constants(self):
        return self._constants

    def get_objective_types(self):
        return self._objective_types

    def get_bounds(self):
        return self._bounds

    def get_num_particles(self):
        return self.num_particles

    def get_num_dimensions(self):
        return self.num_dimensions

    def get_tournament_selection_size(self):
        return self._tournament_selection_size

    def get_extra_bounds(self):
        return self._extra_bounds

    def get_true_pof(self):
        if self._true_pof is None:
            self._true_pof = []
            # extract the true pof from the textfile
            with open("True POF/"+self._current_bench+".pf") as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            for item in content:
                self._true_pof.append(list(map(float, item.split())))
        return self._true_pof


# ----------------------------------------------------------------------------------------------- Benchmark Functions ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- ZDT1 ------------------------------------------------------------ #

    @staticmethod
    def _zdt1_constants():
        return [0.475, 1.80, 1.10, 1.80]

    def zdt1(self):
        self._current_bench = "ZDT1"
        # number of dimensions
        self.num_dimensions = 30

        # PyPi
        zdt1 = zdt.ZDT1()
        f1 = zdt.ZDT1to4_f1
        g = zdt.ZDT1to3_g(self.num_dimensions)
        f2 = zdt.ZDT_f2(f1, g, zdt1.h)

        # objective 1
        # self._objectives.append(self._zdt1_1)
        self._objectives.append(f1)
        # objective 2
        # self._objectives.append(self._zdt1_2)
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('zdt1_1')
        # objective name 2
        self._objective_names.append('zdt1_2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [33, 17]
        # constants
        self._constants = Evaluations._zdt1_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, 1])

# ----------------------------------------------- ZDT2 ------------------------------------------------------------ #

    @staticmethod
    def _zdt2_constants():
        return [0.075, 1.60, 1.35, 1.90]

    def zdt2(self):
        self._current_bench = "ZDT2"
        # number of dimensions
        self.num_dimensions = 30

        # PyPi
        zdt2 = zdt.ZDT2()
        f1 = zdt.ZDT1to4_f1
        g = zdt.ZDT1to3_g(self.num_dimensions)
        f2 = zdt.ZDT_f2(f1, g, zdt2.h)

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('zdt2_1')
        # objective name 2
        self._objective_names.append('zdt2_2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [8, 42]
        # constants
        self._constants = Evaluations._zdt2_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, 1])

# ----------------------------------------------- ZDT3 ------------------------------------------------------------ #

    @staticmethod
    def _zdt3_constants():
        return [0.050, 1.85, 1.90, 1.90]

    def zdt3(self):
        self._current_bench = "ZDT3"
        # number of dimensions
        self.num_dimensions = 30

        # PyPi
        zdt3 = zdt.ZDT3()
        f1 = zdt.ZDT1to4_f1
        g = zdt.ZDT1to3_g(self.num_dimensions)
        f2 = zdt.ZDT_f2(f1, g, zdt3.h)

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('zdt3_1')
        # objective name 2
        self._objective_names.append('zdt3_2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [8, 42]
        # constants
        self._constants = Evaluations._zdt3_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, 1])

# ----------------------------------------------- ZDT4 ------------------------------------------------------------ #

    @staticmethod
    def _zdt4_constants():
        return [0.175, 1.85, 1.35, 1.85]

    def zdt4(self):
        self._current_bench = "ZDT4"
        # number of dimensions
        self.num_dimensions = 10

        # PyPi
        zdt4 = zdt.ZDT4()
        f1 = zdt.ZDT1to4_f1
        g = zdt4.g
        f2 = zdt.ZDT_f2(f1, g, zdt4.h)

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('zdt4_1')
        # objective name 2
        self._objective_names.append('zdt4_2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # lower bound
        self._bounds.append(-5)
        # upper bound
        self._bounds.append(5)
        # number of particles
        self.num_particles = [5, 45]
        # constants
        self._constants = Evaluations._zdt4_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = [[0, 1]]
        for loop in range(1, self.num_dimensions):
            self._bounds.append([-5, 5])

# ----------------------------------------------- ZDT6 ------------------------------------------------------------ #

    @staticmethod
    def _zdt6_constants():
        return [0.600, 1.85, 1.55, 1.80]

    def zdt6(self):
        self._current_bench = "ZDT6"
        # number of dimensions
        self.num_dimensions = 10

        # PyPi
        zdt6 = zdt.ZDT6()
        f1 = zdt6.f1
        g = zdt6.g
        f2 = zdt.ZDT_f2(f1, g, zdt6.h)

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('zdt6_1')
        # objective name 2
        self._objective_names.append('zdt6_2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [1, 49]
        # constants
        self._constants = Evaluations._zdt6_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, 1])


# ----------------------------------------------- WFG1 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg1_2obj_constants():
        return [0.275, 1.65, 1.80, 1.75]

    def wfg1_2obj(self):
        self._current_bench = "WFG1.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg1_2obj_f1
        f2 = Evaluations.__wfg1_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg1_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg1_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [45, 5]
        # constants
        self._constants = Evaluations.__wfg1_2obj_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg1_2obj_f1(decision_variables):

        return wfg.WFG1(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg1_2obj_f2(decision_variables):
        return wfg.WFG1(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG2 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg2_2obj_constants():
        return [0.750, 1.15, 1.70, 1.05]

    def wfg2_2obj(self):
        self._current_bench = "WFG2.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg2_2obj_f1
        f2 = Evaluations.__wfg2_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg2_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg2_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [24, 26]
        # constants
        self._constants = Evaluations.__wfg2_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg2_2obj_f1(decision_variables):
        return wfg.WFG2(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg2_2obj_f2(decision_variables):
        return wfg.WFG2(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG3 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg3_2obj_constants():
        return [0.600, 1.60, 1.85, 0.95]

    def wfg3_2obj(self):
        self._current_bench = "WFG3.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg3_2obj_f1
        f2 = Evaluations.__wfg3_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg3_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg3_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [31, 19]
        # constants
        self._constants = Evaluations.__wfg3_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg3_2obj_f1(decision_variables):
        return wfg.WFG3(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg3_2obj_f2(decision_variables):
        return wfg.WFG3(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG4 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg4_2obj_constants():
        return [0.100, 0.80, 1.65, 1.70]

    def wfg4_2obj(self):
        self._current_bench = "WFG4.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg4_2obj_f1
        f2 = Evaluations.__wfg4_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg4_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg4_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [2, 48]
        # constants
        self._constants = Evaluations.__wfg4_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg4_2obj_f1(decision_variables):
        return wfg.WFG4(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg4_2obj_f2(decision_variables):
        return wfg.WFG4(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG5 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg5_2obj_constants():
        return [0.600, 0.80, 1.60, 1.85]

    def wfg5_2obj(self):
        self._current_bench = "WFG5.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg5_2obj_f1
        f2 = Evaluations.__wfg5_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg5_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg5_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [50, 0]
        # constants
        self._constants = Evaluations.__wfg5_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg5_2obj_f1(decision_variables):
        return wfg.WFG5(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg5_2obj_f2(decision_variables):
        return wfg.WFG5(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG6 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg6_2obj_constants():
        return [0.525, 0.65, 0.60, 1.65]

    def wfg6_2obj(self):
        self._current_bench = "WFG6.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg6_2obj_f1
        f2 = Evaluations.__wfg6_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg6_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg6_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [19, 31]
        # constants
        self._constants = Evaluations.__wfg6_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg6_2obj_f1(decision_variables):
        return wfg.WFG6(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg6_2obj_f2(decision_variables):
        return wfg.WFG6(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG7 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg7_2obj_constants():
        return [0.450, 1.20, 1.85, 1.55]

    def wfg7_2obj(self):
        self._current_bench = "WFG7.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg7_2obj_f1
        f2 = Evaluations.__wfg7_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg7_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg7_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [29, 21]
        # constants
        self._constants = Evaluations.__wfg7_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg7_2obj_f1(decision_variables):
        return wfg.WFG7(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg7_2obj_f2(decision_variables):
        return wfg.WFG7(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG8 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg8_2obj_constants():
        return [0.750, 1.00, 1.65, 1.05]

    def wfg8_2obj(self):
        self._current_bench = "WFG8.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg8_2obj_f1
        f2 = Evaluations.__wfg8_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg8_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg8_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [37, 13]
        # constants
        self._constants = Evaluations.__wfg8_2obj_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg8_2obj_f1(decision_variables):
        return wfg.WFG8(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg8_2obj_f2(decision_variables):
        return wfg.WFG8(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG9 2 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg9_2obj_constants():
        return [0.750, 1.00, 1.65, 1.05]

    def wfg9_2obj(self):
        self._current_bench = "WFG9.2D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg9_2obj_f1
        f2 = Evaluations.__wfg9_2obj_f2

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective name 1
        self._objective_names.append('wfg9_2obj_f1')
        # objective name 2
        self._objective_names.append('wfg9_2obj_f2')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [13, 37]
        # constants
        self._constants = Evaluations.__wfg9_2obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg9_2obj_f1(decision_variables):
        return wfg.WFG9(2, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg9_2obj_f2(decision_variables):
        return wfg.WFG9(2, 24, 4).objective_function(decision_variables)[1]

# ----------------------------------------------- WFG1 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg1_3obj_constants():
        return [0.125, 1.20, 1.30, 1.75]

    def wfg1_3obj(self):
        self._current_bench = "WFG1.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg1_3obj_f1
        f2 = Evaluations.__wfg1_3obj_f2
        f3 = Evaluations.__wfg1_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg1_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg1_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg1_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [37, 4, 9]
        # constants
        self._constants = Evaluations.__wfg1_3obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    # if M >= 3, let k == 2*(M-1) (k == 2*(3-1) == 2*(2) == 4)
    @staticmethod
    def __wfg1_3obj_f1(decision_variables):
        return wfg.WFG1(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg1_3obj_f2(decision_variables):
        return wfg.WFG1(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg1_3obj_f3(decision_variables):
        return wfg.WFG1(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG2 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg2_3obj_constants():
        return [0.275, 1.25, 1.40, 1.70]

    def wfg2_3obj(self):
        self._current_bench = "WFG2.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg2_3obj_f1
        f2 = Evaluations.__wfg2_3obj_f2
        f3 = Evaluations.__wfg2_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg2_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg2_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg2_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [24, 25, 1]
        # constants
        self._constants = Evaluations.__wfg2_3obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg2_3obj_f1(decision_variables):
        return wfg.WFG2(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg2_3obj_f2(decision_variables):
        return wfg.WFG2(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg2_3obj_f3(decision_variables):
        return wfg.WFG2(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG3 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg3_3obj_constants():
        return [0.525, 1.65, 1.75, 0.75]

    def wfg3_3obj(self):
        self._current_bench = "WFG3.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg3_3obj_f1
        f2 = Evaluations.__wfg3_3obj_f2
        f3 = Evaluations.__wfg3_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg3_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg3_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg3_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [29, 10, 11]
        # constants
        self._constants = Evaluations.__wfg3_3obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg3_3obj_f1(decision_variables):
        return wfg.WFG3(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg3_3obj_f2(decision_variables):
        return wfg.WFG3(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg3_3obj_f3(decision_variables):
        return wfg.WFG3(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG4 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg4_3obj_constants():
        return [0.275, 1.75, 0.50, 1.05]

    def wfg4_3obj(self):
        self._current_bench = "WFG4.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg4_3obj_f1
        f2 = Evaluations.__wfg4_3obj_f2
        f3 = Evaluations.__wfg4_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg4_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg4_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg4_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [29, 21, 0]
        # constants
        self._constants = Evaluations.__wfg4_3obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg4_3obj_f1(decision_variables):
        return wfg.WFG4(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg4_3obj_f2(decision_variables):
        return wfg.WFG4(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg4_3obj_f3(decision_variables):
        return wfg.WFG4(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG5 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg5_3obj_constants():
        return [0.575, 0.60, 1.85, 1.75]

    def wfg5_3obj(self):
        self._current_bench = "WFG5.3D"
        # number of dimensions
        self.num_dimensions = 10

        f1 = Evaluations.__wfg5_3obj_f1
        f2 = Evaluations.__wfg5_3obj_f2
        f3 = Evaluations.__wfg5_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg5_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg5_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg5_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [2, 48, 0]
        # constants
        self._constants = Evaluations.__wfg5_3obj_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg5_3obj_f1(decision_variables):
        return wfg.WFG5(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg5_3obj_f2(decision_variables):
        return wfg.WFG5(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg5_3obj_f3(decision_variables):
        return wfg.WFG5(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG6 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg6_3obj_constants():
        return [0.300, 0.90, 0.90, 1.90]

    def wfg6_3obj(self):
        self._current_bench = "WFG6.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg6_3obj_f1
        f2 = Evaluations.__wfg6_3obj_f2
        f3 = Evaluations.__wfg6_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg6_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg6_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg6_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [5, 30, 15]
        # constants
        self._constants = Evaluations.__wfg6_3obj_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg6_3obj_f1(decision_variables):
        return wfg.WFG6(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg6_3obj_f2(decision_variables):
        return wfg.WFG6(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg6_3obj_f3(decision_variables):
        return wfg.WFG6(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG7 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg7_3obj_constants():
        return [0.425, 1.45, 1.50, 1.40]

    def wfg7_3obj(self):
        self._current_bench = "WFG7.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg7_3obj_f1
        f2 = Evaluations.__wfg7_3obj_f2
        f3 = Evaluations.__wfg7_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg7_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg7_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg7_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [10, 22, 18]
        # constants
        self._constants = Evaluations.__wfg7_3obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg7_3obj_f1(decision_variables):
        return wfg.WFG7(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg7_3obj_f2(decision_variables):
        return wfg.WFG7(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg7_3obj_f3(decision_variables):
        return wfg.WFG7(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG8 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg8_3obj_constants():
        return [0.425, 0.95, 1.75, 1.85]

    def wfg8_3obj(self):
        self._current_bench = "WFG8.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg8_3obj_f1
        f2 = Evaluations.__wfg8_3obj_f2
        f3 = Evaluations.__wfg8_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg8_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg8_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg8_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [4, 23, 23]
        # constants
        self._constants = Evaluations.__wfg8_3obj_constants()
        # selection size
        self._tournament_selection_size = 3
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg8_3obj_f1(decision_variables):
        return wfg.WFG8(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg8_3obj_f2(decision_variables):
        return wfg.WFG8(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg8_3obj_f3(decision_variables):
        return wfg.WFG8(3, 24, 4).objective_function(decision_variables)[2]

# ----------------------------------------------- WFG9 3 OBJECTIVE ------------------------------------------------------------ #
    @staticmethod
    def __wfg9_3obj_constants():
        return [0.275, 1.25, 0.75, 1.50]

    def wfg9_3obj(self):
        self._current_bench = "WFG9.3D"
        # number of dimensions
        self.num_dimensions = 24

        f1 = Evaluations.__wfg9_3obj_f1
        f2 = Evaluations.__wfg9_3obj_f2
        f3 = Evaluations.__wfg9_3obj_f3

        # objective 1
        self._objectives.append(f1)
        # objective 2
        self._objectives.append(f2)
        # objective 3
        self._objectives.append(f3)
        # objective name 1
        self._objective_names.append('wfg9_3obj_f1')
        # objective name 2
        self._objective_names.append('wfg9_3obj_f2')
        # objective name 3
        self._objective_names.append('wfg9_3obj_f3')
        # objective type 1
        self._objective_types.append("min")
        # objective type 2
        self._objective_types.append("min")
        # objective type 3
        self._objective_types.append("min")
        # number of particles
        self.num_particles = [4, 45, 1]
        # constants
        self._constants = Evaluations.__wfg9_3obj_constants()
        # selection size
        self._tournament_selection_size = 2
        # bounds
        self._bounds = []
        for loop in range(self.num_dimensions):
            self._bounds.append([0, (2.0 * (loop + 1))])

    @staticmethod
    def __wfg9_3obj_f1(decision_variables):
        return wfg.WFG9(3, 24, 4).objective_function(decision_variables)[0]

    @staticmethod
    def __wfg9_3obj_f2(decision_variables):
        return wfg.WFG9(3, 24, 4).objective_function(decision_variables)[1]

    @staticmethod
    def __wfg9_3obj_f3(decision_variables):
        return wfg.WFG9(3, 24, 4).objective_function(decision_variables)[2]


