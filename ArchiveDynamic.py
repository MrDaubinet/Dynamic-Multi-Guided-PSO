import random
import copy
import math
import os


class ArchiveDynamic:
    def __init__(self, num_particles, evaluation, algorithm_name, archive_strategy, dimensionality):
        self.archive_particles = []
        self.max_size = num_particles
        self.sorted_objective_values = []
        self.evaluation = evaluation
        self.__generational_distances = []
        self.__archive_count = 0
        self.algorithm_name = algorithm_name
        self.archive_strategy = archive_strategy
        self.dimensionality = dimensionality

# ------------------------------------------------------# Public #----------------------------------------------------#
    def add_to_archive(self, particle):
        """
        Add a particle to the archive.
        :param particle: particle object
        :return: None
        """
        # add required objective information (for crowding distance)
        self.__add_objective_values(particle)
        # check if the solution is feasible
        if self.__feasible(particle):
            # check if the solution is not dominated by anything in the archive
            if not self.__check_if_dominated(particle):
                self.__remove_dominating(particle)
                # # check if the archive is full
                # if self.__check_if_fill():
                #     self.__remove_worst_solution()
                self.archive_particles.append(copy.deepcopy(particle))
                self.__set_crowding_distance()
                # check if the archive is full
                if self.__check_if_fill():
                    self.__remove_worst_solution()
        return

    def get_guide(self):
        # select 3 random solutions from the archive
        max_distance = float('-inf')
        tournament_index = -1
        if len(self.archive_particles) <= self.evaluation.get_tournament_selection_size():
            tournament = self.archive_particles
        else:
            tournament = random.sample(self.archive_particles, self.evaluation.get_tournament_selection_size())

        for particle_index in range(len(tournament)):
            crowding_distance = sum(tournament[particle_index].crowding_distances)
            if float(crowding_distance) > max_distance:
                max_distance = crowding_distance
                tournament_index = particle_index
        if tournament_index == -1:
            return None
        else:
            return tournament[tournament_index]

    def get_archive(self):
        return self.archive_particles

    def refresh_archive(self):
        # save the archive
        self.__save_archive()
        # clear archive
        temp_archive = copy.deepcopy(self.archive_particles)
        self.archive_particles = []
        # re Add all particles
        for particle in temp_archive:
            self.add_to_archive(copy.deepcopy(particle))

    def save_the_archive(self):
        self.__save_archive()

    def reinitialize_archive(self):
        self.__save_archive()
        self.archive_particles = []

# ----------------------------------------------------# Private #---------------------------------------------------- #

    def __check_if_dominated(self, particle):
        """Check if the particle to be added to the archive is dominated by any particles in the archive"""
        # A DOMINATED point is one in which there exist at least one solution in the archive which is better
        # than this point for all objectives
        for particle_index in range(len(self.archive_particles)):
            dominated_count = 0
            better_than_count = 0
            for objective_index in range(len(self.archive_particles[particle_index].objective_values)):
                if self.evaluation.get_objective_types()[objective_index] == 'min':
                    if self.archive_particles[particle_index].objective_values[objective_index] <= particle.objective_values[objective_index]:
                        dominated_count += 1
                    if self.archive_particles[particle_index].objective_values[objective_index] < particle.objective_values[objective_index]:
                        better_than_count += 1
                else:
                    if self.archive_particles[particle_index].objective_values[objective_index] >= particle.objective_values[objective_index]:
                        dominated_count += 1
                    if self.archive_particles[particle_index].objective_values[objective_index] > particle.objective_values[objective_index]:
                        better_than_count += 1
            if dominated_count == len(particle.objective_values) and better_than_count > 0:
                return True
        return False

    def __set_crowding_distance(self):
        for objective_index in range(len(self.evaluation.get_objective_functions())):
            self.__sort_archive_by_objective(objective_index)
            self.__objective_crowding_distance(objective_index)

    def __remove_dominating(self, particle):
        remove_indexes = []
        for particle_index in range(len(self.archive_particles)):
            dominated_count = 0
            greater_than_count = 0
            for objective_index in range(len(self.archive_particles[particle_index].objective_values)):
                if self.evaluation.get_objective_types()[objective_index] == 'min':
                    if particle.objective_values[objective_index] <= self.archive_particles[particle_index].objective_values[objective_index]:
                        dominated_count += 1
                    if particle.objective_values[objective_index] < self.archive_particles[particle_index].objective_values[objective_index]:
                        greater_than_count += 1
                else:
                    if particle.objective_values[objective_index] >= self.archive_particles[particle_index].objective_values[objective_index]:
                        dominated_count += 1
                    if particle.objective_values[objective_index] > self.archive_particles[particle_index].objective_values[objective_index]:
                        greater_than_count += 1
            if dominated_count == len(particle.objective_values) and greater_than_count > 0:
                # store the indexes to be removed
                remove_indexes.append(particle_index)
        for index in reversed(remove_indexes):
            # remove_index = index-removed_count
            del self.archive_particles[index]
        return

    def __check_if_fill(self):
        if len(self.archive_particles) > self.max_size:
            return True
        else:
            return False

    def __remove_worst_solution(self):
        # find the solution which has the greatest hamming distance and remove it from the archive

        # Search for the solution with the smallest crowding distance
        min_distance = float('inf')
        archive_index = -1
        for particle_index in range(len(self.archive_particles)):
            crowding_distance = sum(self.archive_particles[particle_index].crowding_distances)
            if crowding_distance < min_distance:
                min_distance = crowding_distance
                archive_index = particle_index

        # remove the particle which has the smallest crowding distance from the archive
        del self.archive_particles[archive_index]
        return

    def __add_objective_values(self, particle):
        # for each objective value hold a list of indexes which point to solutions in the archive
        objective_values = []
        particle.crowding_distances = []
        for objective_function in self.evaluation.get_objective_functions():
            objective_values.append(objective_function(particle.position_indexes, self.evaluation.get_t(), self.evaluation.get_r_i()))
            particle.crowding_distances.append(-1)
        particle.objective_values = objective_values
        return

    def __sort_archive_by_objective(self, objective_index):
        # for each particle in the archive, bubble sort it by its objective_values[objective_index]
        for particle_index in range(len(self.archive_particles)):
            for particle_index_2 in range(0, len(self.archive_particles)-particle_index-1):
                if self.archive_particles[particle_index_2].objective_values[objective_index] \
                        > self.archive_particles[particle_index_2+1].objective_values[objective_index]:
                    self.archive_particles[particle_index_2], self.archive_particles[particle_index_2+1] = \
                        self.archive_particles[particle_index_2+1], self.archive_particles[particle_index_2]
        return

    def __objective_crowding_distance(self, objective_index):
        # for each particle in the archive
        for particle_index in range(len(self.archive_particles)):
            if particle_index == 0 or particle_index == (len(self.archive_particles)-1):
                self.archive_particles[particle_index].crowding_distances[objective_index] = float('inf')
            else:
                objective_max = self.archive_particles[len(self.archive_particles) - 1].objective_values[objective_index]
                objective_min = self.archive_particles[0].objective_values[objective_index]
                distance = abs(self.archive_particles[particle_index+1].objective_values[objective_index] -
                        self.archive_particles[particle_index-1].objective_values[objective_index])

                if objective_max - objective_min != 0:
                    crowding_distance = distance/(objective_max - objective_min)
                else:
                    crowding_distance = 0
                self.archive_particles[particle_index].crowding_distances[objective_index] = crowding_distance

        return

    @staticmethod
    def __feasible(particle):
        for i in range(0, particle.num_dimensions):
            # check maximum bounds
            if particle.position_indexes[i] > particle.bounds[i][1]:
                return False

            # check minimum bounds
            if particle.position_indexes[i] < particle.bounds[i][0]:
                return False
        return True

    def __save_archive(self):
        objective_names = self.evaluation.get_objective_names()
        for objective_index in range(len(objective_names)):
            file_directory_pof = "Dynamic POF/"+self.algorithm_name+"/"+self.archive_strategy+"/"+self.dimensionality+"/nT_"+str(self.evaluation.severity_of_change)+"_tT_"+str(self.evaluation.frequency_of_change)+"/"+objective_names[objective_index]+"/run_"+str(self.evaluation.get_run())+"/archive_"+str(self.__archive_count)
            if not os.path.exists(os.path.dirname(file_directory_pof)):
                os.makedirs(os.path.dirname(file_directory_pof))
            file_writer = open(file_directory_pof, 'w')
            file_writer.close()
        for particle_index in range(len(self.archive_particles)):
            for objective_index in range(len(self.archive_particles[particle_index].objective_values)):
                file_directory_pof = "Dynamic POF/" + self.algorithm_name + "/" + self.archive_strategy + "/" + self.dimensionality + "/nT_" + str(self.evaluation.severity_of_change) + "_tT_" + str(
                    self.evaluation.frequency_of_change) + "/" + objective_names[objective_index] + "/run_" + str(self.evaluation.get_run()) + "/archive_" + str(self.__archive_count)
                file_writer = open(file_directory_pof, 'a')
                file_writer.write("%s\n" % self.archive_particles[particle_index].objective_values[objective_index])
                file_writer.close()
        # update the archive count
        self.__archive_count += 1
        return

    def __generational_distance(self):
        """calculate the Generational distance:
        for each solution in the Archive, find the solution in the true POF which is closest to it."""
        true_pof = self.evaluation.get_true_pof()
        generational_distance = 0
        # for each particle in the archive
        for particle_index in range(len(self.archive_particles)):
            closest_distance = 100000
            # find the closest pof
            for row in true_pof:
                euclidean_distance = 0
                for objective_index in range(len(self.archive_particles[particle_index].objective_values)):
                    euclidean_distance += math.fabs(row[objective_index] - self.archive_particles[particle_index].objective_values[objective_index])
                euclidean_distance = math.pow(euclidean_distance, 2)
                if euclidean_distance < closest_distance:
                    closest_distance = euclidean_distance
            # calculate the distance : |pof - true POF|
            generational_distance += closest_distance

        generational_distance = math.sqrt(generational_distance) / len(self.archive_particles)
        self.__generational_distances.append(generational_distance)
        return



