import numpy as np
from datetime import datetime as dt


# 유전 알고리즘
class GeneticAlgorithm:
    def __init__(self, problem, epochs, num_of_chromosome, num_of_gene, target_fitness=None, plot_interval=0,
                 mutation_mode='reverse', mutation_rate=0.001, max_mutation_rate=1.):
        self.problem = problem

        self.epochs = epochs
        self.target_fitness = target_fitness
        self.num_of_chromosome = num_of_chromosome
        self.num_of_gene = num_of_gene
        self.mutation_mode = mutation_mode
        self.init_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.plot_interval = plot_interval

    def initialize_chromosome(self):
        # TSP 점의 갯수 -1 을 유전자로 갖는 염색체 n 개 생성.
        chromosomes = [np.arange(1, self.num_of_gene + 1) for _ in range(self.num_of_chromosome)]

        for chromosome in chromosomes:
            np.random.shuffle(chromosome)

        return np.array(chromosomes)

    @staticmethod
    def softmax(a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def generate_offspring(self, chromosomes, fitness):
        # 가장 좋은 염색체는 보존.
        offsprings = [chromosomes[np.argmax(fitness)]]

        # 적합도에 음수가 있을 수 있기 때문에 softmax 사용.
        p = self.softmax(fitness)
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

            # 돌연변이 (mutation) 연산. exchange
            if np.random.random() < self.mutation_rate:
                indices = np.random.randint(self.num_of_gene, size=2)

                if self.mutation_mode == 'reverse':
                    low = min(indices)
                    high = max(indices)
                    offspring[low:high] = np.flip(offspring[low:high])
                elif self.mutation_mode == 'exchange':
                    offspring[indices[0]], offspring[indices[1]] = offspring[indices[1]], offspring[indices[0]]

            offsprings.append(offspring)

        return np.array(offsprings)

    def solve(self):
        # 1. 초기 염색체 집합 생성 (chromosome initialize)
        chromosomes = self.initialize_chromosome()

        # 2. 초기 염색체들에 대한 적합도 계산
        fitness = self.problem.calc_fitness(chromosomes)

        best_fitness = np.max(fitness)

        print('[{}] init. max fitness: {:.15f}'.format(dt.now(), best_fitness))

        for e in range(self.epochs):
            if self.plot_interval and e % self.plot_interval == 0:
                self.problem.render(pause=0.3)

            if self.target_fitness and self.target_fitness <= best_fitness:
                break

            # 3. 현재 염색체들로부터 자손들을 생성
            chromosomes = self.generate_offspring(chromosomes, fitness)

            # 4. 생성된 자손들의 적합도 계산
            fitness = self.problem.calc_fitness(chromosomes)

            if best_fitness < np.max(fitness):
                best_fitness = np.max(fitness)
                self.mutation_rate = self.init_mutation_rate
            else:
                self.mutation_rate = min(self.mutation_rate * 2, self.max_mutation_rate)

            print('[{}] {:3} epochs. max fitness: {:.15f} mutation rate: {}'.format(dt.now(),
                                                                                    e,
                                                                                    best_fitness,
                                                                                    self.mutation_rate))

        print('[{}] max fitness: {:.15f}'.format(dt.now(),best_fitness))
        print('solution: {}'.format(chromosomes[np.argmax(fitness)]))
