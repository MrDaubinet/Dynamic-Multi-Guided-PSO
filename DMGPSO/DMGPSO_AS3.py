from DMGPSO import Particle
import DynamicArchive
import copy


class PSODynamic:
    def __init__(self, max_iterations, dynamic_evaluations):
        """
        objective_functions: objective function array
        bounds: Bounds array (bounds for each objective value)
        objective_types: array of objective type (min or max)
        num_particles: Array of the number of particles for each swarm
        max_iterations: Number of iterations
        dimensions: Number of dimensions (length of particle position vector)
        """
        best_swarm_global_fitness_values = []        # best error for group
        swarm_gbest_positions = []                   # best position for group
        evaluations = dynamic_evaluations
        objective_functions = evaluations.get_objective_functions()
        num_particles = evaluations.get_num_particles()
        archive = DynamicArchive.ArchiveDynamic(sum(num_particles), evaluations, "MGPSO", "Archive Strategy 3", evaluations.get_dimensions_type())
        constants = evaluations.get_constants()
        objective_types = evaluations.get_objective_types()
        dimensions = evaluations.get_num_dimensions()
        bounds = evaluations.get_bounds()

        for objective_index in range(len(objective_functions)):
            if objective_types[objective_index] == "min":
                best_swarm_global_fitness_values.append(float('inf'))
            else:
                best_swarm_global_fitness_values.append(float('-inf'))
            swarm_gbest_positions.append([])

        # for each objective make a swarm
        swarms = []
        for objective_index in range(len(objective_functions)):
            # establish the swarm
            swarm = []
            for particle in range(0, num_particles[objective_index]):
                swarm.append(
                    particle.ParticleDynamic(dimensions, objective_types[objective_index], bounds, constants[0], constants[1], constants[2], constants[3]))
            swarms.append(copy.deepcopy(swarm))

        # begin optimization loop

        iteration = 1
        evaluations.update_t(iteration)
        prev_t = evaluations.get_t()
        while iteration <= max_iterations:
            # print("iteration: " + str(iteration))
            evaluations.update_t(iteration)
            # check if t has changed
            if evaluations.get_t() != prev_t:
                # reset particle pbest and gbest values
                PSODynamic.reset_particle(swarms)
                # reset best_swarm_global_fitness_values
                # reset swarm_gbest_positions
                best_swarm_global_fitness_values = []  # best error for group
                swarm_gbest_positions = []
                for objective_index in range(len(objective_functions)):
                    if objective_types[objective_index] == "min":
                        best_swarm_global_fitness_values.append(float('inf'))
                    else:
                        best_swarm_global_fitness_values.append(float('-inf'))
                    swarm_gbest_positions.append([])
                # refresh archive
                archive.reinitialize_archive()
                # update prev_t
                prev_t = evaluations.get_t()

            for objective_index in range(len(objective_functions)):
                # cycle through particles in objective swarm and evaluate fitness
                for particle_index in range(0, num_particles[objective_index]):
                    swarms[objective_index][particle_index].evaluate(objective_functions[objective_index], evaluations.get_t(), evaluations.get_r_i())
                    # check to see if the current position is an individual best
                    if (objective_types[objective_index] == "min" and swarms[objective_index][particle_index].fitness_function_value < swarms[objective_index][particle_index].best_fitness_value) \
                            or (objective_types[objective_index] == "max" and swarms[objective_index][particle_index].fitness_function_value > swarms[objective_index][particle_index].best_fitness_value) \
                            or swarms[objective_index][particle_index].best_fitness_value == -1:
                        swarms[objective_index][particle_index].pbest_position_indexes = copy.deepcopy(swarms[objective_index][particle_index].position_indexes)
                        swarms[objective_index][particle_index].best_fitness_value = float(swarms[objective_index][particle_index].fitness_function_value)

                    # determine if current particle is the best (globally) in its swarm
                    if (objective_types[objective_index] == "min" and swarms[objective_index][particle_index].best_fitness_value < best_swarm_global_fitness_values[objective_index])\
                            or (objective_types[objective_index] == "max" and swarms[objective_index][particle_index].best_fitness_value > best_swarm_global_fitness_values[objective_index])\
                            or best_swarm_global_fitness_values[objective_index] == -1:
                        swarm_gbest_positions[objective_index] = copy.deepcopy(swarms[objective_index][particle_index].pbest_position_indexes)
                        best_swarm_global_fitness_values[objective_index] = float(swarms[objective_index][particle_index].best_fitness_value)

                    # update the archive with the solution
                    archive.add_to_archive(swarms[objective_index][particle_index])

            # for each objective
            for objective_index in range(len(objective_functions)):
                # cycle through swarm and update velocities and position
                for particle_index in range(0, num_particles[objective_index]):
                    # get the guide particle
                    guide_particle = archive.get_guide()
                    swarms[objective_index][particle_index].update_velocity(swarm_gbest_positions[objective_index], guide_particle)
                    swarms[objective_index][particle_index].update_position()
            iteration += 1

        # print final results
        archive.save_archive_to_file()
        print('Completed')

    @staticmethod
    def reset_particle(swarms):
        """
        for every particle in each swarm,
            # reset best position
            # reset best fitness value
        Reset best_swarm_global_fitness_values.
        Reset swarm_gbest_positions.
        :param swarms:
        :return:
        """
        # for every particle in each swarm,
        # reset best position
        # reset best fitness value
        for swarm_index in range(len(swarms)):
            for particle_index in range(len(swarms[swarm_index])):
                swarms[swarm_index][particle_index].pbest_position_indexes = []
                if swarms[swarm_index][particle_index].objective_type == "min":
                    swarms[swarm_index][particle_index].best_fitness_value = float('inf')
                else:
                    swarms[swarm_index][particle_index].best_fitness_value = float('-inf')
                # possibly reset velocity,+
                # swarms[swarm_index][particle_index].velocity_indexes = [0] * swarms[swarm_index][particle_index].num_dimensions
        return

