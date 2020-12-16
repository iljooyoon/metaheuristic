import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt


# 유전 알고리즘
class GeneticAlgorithm:
    def __init__(self, num_of_chromosome, num_of_gene, mutation_rate=0.001, max_mutation_rate = 1., plot_interval=0):
        self.num_of_chromosome = num_of_chromosome
        self.num_of_gene = num_of_gene
        self.init_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.plot_interval = plot_interval

    def initialize_chromosome(self):
        chromosomes = [np.arange(1, self.num_of_gene + 1) for _ in range(self.num_of_chromosome)]

        for chromosome in chromosomes:
            np.random.shuffle(chromosome)

        return np.array(chromosomes)

    def index2point(self, problem, c):
        full_idx = np.concatenate((np.array([0] * self.num_of_chromosome)[:, None], c), axis=1)
        return np.take_along_axis(problem[None].repeat(self.num_of_chromosome, axis=0),
                                  indices=full_idx[:, :, None].repeat(2, axis=2),
                                  axis=1)

    @staticmethod
    def calc_fitness(points):
        shifted = np.roll(points, shift=1, axis=1)
        dist = np.sqrt(((points - shifted) ** 2).sum(axis=2)).sum(axis=1)
        return -dist
    # def calc_fitness(points):
    #     shifted = np.roll(points, shift=1, axis=1)
    #     dist = np.sqrt(((points - shifted) ** 2).sum(axis=2)).sum(axis=1)
    #     return (1000/dist)**2

    def generate_offspring(self, chromosomes, fitness):
        # 가장 좋은 염색체는 보존.
        offsprings = [chromosomes[np.argmax(fitness)]]

        # 적합도에 음수가 있을 수 있기 때문에 softmax 사용.
        p = softmax(fitness)
        # 룰렛 휠 방식 가능
        # p = fitness / sum(fitness)

        for _ in range(self.num_of_chromosome - 1):
            # 부모 염색체 선택
            parents = chromosomes[np.random.choice(self.num_of_chromosome, size=2, replace=False, p=p)]

            # 부모로 부터 자손 생성
            division_point = np.random.randint(self.num_of_gene)
            offspring = parents[0][:division_point]
            for gene in parents[1]:
                if gene not in offspring:
                    offspring = np.concatenate((offspring, gene[None]), axis=0)

            # 0.1% 확률로 돌연변이 (mutation) 연산. exchange
            if np.random.random() < self.mutation_rate:
                indexes = np.random.randint(self.num_of_gene, size=2)
                offspring[indexes[0]], offspring[indexes[1]] = offspring[indexes[1]], offspring[indexes[0]]

            offsprings.append(offspring)

        return np.array(offsprings)

    def solve(self, problem, epoch, target_fitness=None):
        # 1. 초기 염색체 집합 생성 (chromosome initialize)
        # 20개(19개로 해도 되지만) 점을 유전자로 갖는 염색체 n 개 생성.
        chromosomes = self.initialize_chromosome()

        # 2. 초기 염색체들에 대한 적합도 계산
        # 적합도 계산 함수 생성. (적합도는 score 와 동일하게 해도 되지 않을까? -거리. 음수가 안된다면 역수 취하기. 혹은 mean을 뺀 값.)
        # 적합도 계산.
        solutions = self.index2point(problem, chromosomes)
        fitness = self.calc_fitness(solutions)

        print('[{}] init. max fitness: {:.15f}'.format(dt.now(), np.max(fitness)))

        best_fitness = np.max(fitness)

        for e in range(epoch):
            if self.plot_interval and e % self.plot_interval == 0:
                plot(problem, solutions[np.argmax(fitness)], pause=0.3)

            if target_fitness and target_fitness <= max(fitness):
                break

            # 3. 현재 염색체들로부터 자손들을 생성
            # 현재 부모 중 가장 best 는 그대로 하나 자손으로 유지 하면 좋을듯.
            # softmax 사용가능. 부모 선택 -> 자손 생성 -> 돌연변이 생성
            chromosomes = self.generate_offspring(chromosomes, fitness)

            # 4. 생성된 자손들의 적합도 계산
            solutions = self.index2point(problem, chromosomes)
            fitness = self.calc_fitness(solutions)

            if best_fitness < np.max(fitness):
                best_fitness = np.max(fitness)
                self.mutation_rate = self.init_mutation_rate
            else:
                self.mutation_rate = min(self.mutation_rate * 2, self.max_mutation_rate)

            if self.plot_interval and e % self.plot_interval == self.plot_interval - 1:
                plt.close('all')

            print('[{}] {:3} epochs. max fitness: {:.15f} mutation rate: {}'.format(dt.now(),
                                                                                    e,
                                                                                    np.max(fitness),
                                                                                    self.mutation_rate))

        return solutions, chromosomes, fitness


def get_distance(ans):
    shifted = np.roll(ans, 1, 0)
    return np.sqrt(((ans - shifted)**2).sum(axis=1)).sum()


def plot(problem, ans, pause=-1.):
    fig = plt.figure(figsize=(10, 7))
    # plt.get_current_fig_manager().window.setGeometry()
    ax = fig.add_subplot(121)
    ax.set_aspect('equal', adjustable='box')

    x, y = np.hsplit(problem, 2)
    ax.plot(x.squeeze()[0], y.squeeze()[0], 'ro')
    ax.plot(x.squeeze()[1:], y.squeeze()[1:], 'o')

    ax = fig.add_subplot(122)
    ax.set_aspect('equal', adjustable='box')
    dist_1 = get_distance(ans)

    sol_loop = np.concatenate((ans, ans[[0]]), axis=0)
    ax.plot(sol_loop[:, 0], sol_loop[:, 1], 'r-')
    ax.plot(ans[0, 0], ans[0, 1], 'ro')
    ax.plot(ans[1:, 0], ans[1:, 1], 'ko')

    ax.set_title('{}'.format(dist_1))
    ax.set_aspect('equal', 'box')

    fig.tight_layout()

    if hasattr(plt.get_current_fig_manager().window, 'wm_geometry'):
        plt.get_current_fig_manager().window.wm_geometry("+500+50")
    elif hasattr(plt.get_current_fig_manager().window, 'SetPosition'):
        plt.get_current_fig_manager().window.SetPosition((500, 50))

    if pause > 0:
        plt.pause(pause)
    else:
        plt.show()


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def main(point_num):
    # TSP 문제 생성. (점 20개 범위 0~1 사이)
    problem = np.random.random((point_num, 2))

    # 유전 알고리즘 실행
    ga = GeneticAlgorithm(num_of_chromosome=50000, num_of_gene=point_num-1, plot_interval=1, mutation_rate=0.01,
                          max_mutation_rate=0.5)
    answers, index_orders, scores = ga.solve(problem=problem, epoch=500)

    print('[{}] max score: {:.15f}'.format(dt.now(), np.max(scores)))

    ans = answers[np.argmax(scores)]

    # plot
    plot(problem, ans)


if __name__ == '__main__':
    main(point_num=50)
