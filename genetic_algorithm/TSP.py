import numpy as np
import matplotlib.pyplot as plt

from GeneticAlgorithm import GeneticAlgorithm


class TSP:
    def __init__(self, batch_size, num_of_points):
        self.batch_size = batch_size
        self.problems = np.random.random((num_of_points, 2))[None].repeat(self.batch_size, axis=0)
        self.solutions = None
        self.fitness = None

    def calc_fitness(self, indices):
        self.solutions = np.concatenate((np.zeros((self.batch_size, 1), dtype=np.int32), indices), axis=1)

        self.fitness = -self.get_distance()

        return self.fitness

    def get_distance(self):
        points = self.index2point(self.problems, self.solutions)

        shifted = np.roll(points, shift=1, axis=1)
        dist = np.sqrt(((points - shifted) ** 2).sum(axis=2)).sum(axis=1)

        return dist

    @staticmethod
    def index2point(problems, indices):
        return np.take_along_axis(problems,
                                  indices=np.expand_dims(indices, -1).repeat(2, axis=-1),
                                  axis=-2)

    def render(self, filename=None, pause=-1.):
        plt.close('all')

        problem = self.problems[0]

        best_solution_idx = self.solutions[np.argmax(self.fitness)]
        best_solution = self.index2point(problem, best_solution_idx)

        fig = plt.figure(figsize=(10, 7))

        ax = fig.add_subplot(121)
        ax.set_aspect('equal', adjustable='box')

        x, y = np.hsplit(problem, 2)
        ax.plot(x.squeeze()[0], y.squeeze()[0], 'ro')
        ax.plot(x.squeeze()[1:], y.squeeze()[1:], 'o')

        ax = fig.add_subplot(122)
        ax.set_aspect('equal', adjustable='box')
        dist_1 = max(self.fitness)

        sol_loop = np.concatenate((best_solution, best_solution[[0]]), axis=0)
        ax.plot(sol_loop[:, 0], sol_loop[:, 1], 'r-')
        ax.plot(best_solution[0, 0], best_solution[0, 1], 'ro')
        ax.plot(best_solution[1:, 0], best_solution[1:, 1], 'ko')

        ax.set_title('{}'.format(dist_1))
        ax.set_aspect('equal', 'box')

        fig.tight_layout()

        if filename:
            fig.savefig('{}.jpg'.format(filename))
        else:
            if hasattr(plt.get_current_fig_manager().window, 'wm_geometry'):
                plt.get_current_fig_manager().window.wm_geometry("+500+50")
            elif hasattr(plt.get_current_fig_manager().window, 'SetPosition'):
                plt.get_current_fig_manager().window.SetPosition((500, 50))

            if pause > 0:
                plt.pause(pause)
            else:
                plt.show()


if __name__ == '__main__':
    chromosome = 50000
    points = 50
    epochs = 500
    tsp = TSP(batch_size=chromosome, num_of_points=points)
    GeneticAlgorithm(tsp, epochs=epochs, num_of_chromosome=chromosome, num_of_gene=points-1, plot_interval=1,
                     mutation_rate=0.01, max_mutation_rate=0.5).solve()
