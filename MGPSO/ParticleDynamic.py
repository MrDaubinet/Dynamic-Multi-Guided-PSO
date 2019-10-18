import random


class ParticleDynamic:
    def __init__(self, num_dimensions, objective_type, bounds, w, c1, c2, c3):
        self.position_indexes = []          # particle position
        self.velocity_indexes = []          # particle velocity
        self.pbest_position_indexes = []     # best position individual
        self.best_fitness_value = None  # best error individual
        self.bounds = bounds
        self.num_dimensions = num_dimensions

        self.fitness_function_value = None
        self.guide_influence = random.random()
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.objective_type = objective_type

        if objective_type == "min":
            self.best_fitness_value = float('inf')
        else:
            self.best_fitness_value = float('-inf')

        for i in range(0, num_dimensions):
            self.velocity_indexes.append(0)
            self.position_indexes.append(random.uniform(bounds[i][0], bounds[i][1]))

    # evaluate current fitness
    def evaluate(self, objective_function, t, r_i):
        self.fitness_function_value = objective_function(self.position_indexes, t, r_i)
        return

    # update new particle velocity
    def update_velocity(self, gbest_position_indexes, archive_guide):
        """
        w = 0.729844       # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1.49618       # cognitive constant
        c2 = 1.49618       # social constant
        c3 = 1.49618       # guide constant"""

        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()

            vel_cognitive = self.c1 * r1 * (self.pbest_position_indexes[i] - self.position_indexes[i])
            vel_social = self.guide_influence*self.c2*r2*(gbest_position_indexes[i] - self.position_indexes[i])
            vel_guide = (1 - self.guide_influence) * self.c3 * r3 * (archive_guide.position_indexes[i] - self.position_indexes[i])

            self.velocity_indexes[i] = self.w * self.velocity_indexes[i] + vel_cognitive + vel_social + vel_guide
        return

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0, self.num_dimensions):
            self.position_indexes[i] = self.position_indexes[i] + self.velocity_indexes[i]

            # adjust maximum position if necessary
            if self.position_indexes[i] > self.bounds[i][1]:
                self.position_indexes[i] = self.bounds[i][1]

            # adjust minimum position if necessary
            if self.position_indexes[i] < self.bounds[i][0]:
                self.position_indexes[i] = self.bounds[i][0]

        return
