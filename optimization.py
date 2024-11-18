import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import distance

# 재난 정보와 병합된 CSV 파일 경로
file_path = r"C:\Users\김소민\Desktop\사문\빅콘\disaster_tr.csv"  
coa= pd.read_csv(file_path)  
coa = coa.drop_duplicates(subset=['STR_LA', 'STR_LO'], keep='first')
coa


# 유전 알고리즘을 위한 초기 설정
population_size = 50  # 개체군 크기
generations = 100  # 세대 수
mutation_rate = 0.1  # 돌연변이 확률

# 입력 값
radius = 10  # 반경 (km)
max_capacity = 50  # 최대 수거량
weights = [0.7, 0.3, 0]  # IEM_CNT, fish, flood/typoon 가중치


def fitness(chromosome, data, radius, max_capacity, weights):
    total_trash = 0
    total_fish = 0
    disaster_score = 0
    covered_indices = set()

    for idx in chromosome:
        row = data.iloc[idx]
        distances = distance.cdist([[row['STR_LA'], row['STR_LO']]], data[['STR_LA', 'STR_LO']].values).flatten()
        nearby_indices = np.where(distances <= radius)[0]

        for nearby_idx in nearby_indices:
            if nearby_idx not in covered_indices:
                covered_indices.add(nearby_idx)
                distance_factor = 1 - (distances[nearby_idx] / radius)  # 거리 기반 가중치
                total_trash += data.iloc[nearby_idx]['IEM_CNT'] * distance_factor
                total_fish += data.iloc[nearby_idx]['fish'] * distance_factor
                disaster_score += data.iloc[nearby_idx][['flood', 'typoon']].sum() * distance_factor

    # 최대 수거량 제한
    if total_trash > max_capacity:
        reduction_factor = max_capacity / total_trash
        total_trash *= reduction_factor
        total_fish *= reduction_factor
        disaster_score *= reduction_factor

    # 가중치 적용 후 점수 계산
    score = (
        weights[0] * total_trash +
        weights[1] * total_fish +
        weights[2] * disaster_score
    )
    return score, total_trash




# 유전 알고리즘 수정 필요 없음

# 초기 개체군 생성
def initialize_population(data, population_size):
    return [np.random.choice(len(data), 3, replace=False) for _ in range(population_size)]

# 교배 연산
def crossover(parent1, parent2):
    # 교배가 가능한 경우에만 수행
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1  # 부모를 그대로 반환

    # 유효한 교배 포인트 설정
    crossover_point = np.random.randint(1, min(len(parent1), len(parent2)))
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return np.unique(child)[:3]  # 중복 제거 후 자식 반환


# 돌연변이 연산
def mutate(chromosome, data_size, mutation_rate):
    if np.random.rand() < mutation_rate:
        chromosome[np.random.randint(0, len(chromosome))] = np.random.randint(0, data_size)
    return chromosome

def genetic_algorithm(data, radius, max_capacity, weights, population_size, generations, mutation_rate):
    population = initialize_population(data, population_size)
    best_total_trash = 0
    best_locations = None

    for generation in range(generations):
        print(f"Generation {generation}:")

        fitness_scores = []
        for idx, chromosome in enumerate(population):
            try:
                score, total_trash = fitness(chromosome, data, radius, max_capacity, weights)
                fitness_scores.append((score, total_trash))
                print(f"  Chromosome {idx}: {chromosome}, Fitness Score: {score}, Total Trash: {total_trash}")
            except Exception as e:
                print(f"  Error with Chromosome {idx}: {chromosome}, Error: {e}")
                raise

        # 최적 쓰레기량 업데이트
        max_trash_entry = max(fitness_scores, key=lambda x: x[1])  # Total trash 기준
        if max_trash_entry[1] > best_total_trash:
            best_total_trash = max_trash_entry[1]
            best_locations = population[fitness_scores.index(max_trash_entry)]

        # 정렬 및 다음 세대 선택
        fitness_scores.sort(reverse=True, key=lambda x: x[0])  # Fitness Score 기준
        next_generation = [population[idx] for idx in range(population_size // 2)]

        # 교배 및 돌연변이
        try:
            while len(next_generation) < population_size:
                parents = np.random.choice(len(next_generation), 2, replace=False)
                child = crossover(next_generation[parents[0]], next_generation[parents[1]])
                if len(child) < 3:
                    continue
                child = mutate(child, len(data), mutation_rate)
                next_generation.append(child)
        except Exception as e:
            print("Error during crossover or mutation:", e)
            raise

        population = next_generation

    # 최적 해 반환
    return data.iloc[best_locations], best_total_trash



# 실행
optimized_locations, total_trash = genetic_algorithm(
    data=coa,
    radius=radius,
    max_capacity=max_capacity,
    weights=weights,
    population_size=population_size,
    generations=generations,
    mutation_rate=mutation_rate
)

print("\n최적 위치:")
print(optimized_locations)
print(f"\n총 수거 가능한 쓰레기량: {total_trash}")

