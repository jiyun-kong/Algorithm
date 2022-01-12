# 서로소 집합 (Disjoint Sets) : 공통 원소가 없는 두 집합
# ex. {1, 2}, {3, 4}

# 서로소 집합 자료구조 : 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조
# 서로소 집합 자료구조는 두 종류의 연산을 지원한다.
# : 합집합 (union) - 두개의 원소가 포함된 집합을 하나의 집합으로 합치는 연산,
# : 찾기 (find) - 특정한 우너소가 속한 집합이 어떤 집합인지 알려주는 연산
# 서로소 집합 자료구조는 합치기 찾기 (Union Find) 자료구조라고 불리기도 한다.

# 여러 개의 합치기 연산이 주어졌을 때 서로소 집합 자료구조의 동작 과정은 다음과 같다.
# 1. 합집합 연산을 확인하여, 서로 연결된 두 노드 A, B를 확인한다.
# 1-1. A와 B의 루트 노드 A', B'를 각각 찾는다.
# 1-2. A'을 B'의 부모 노드로 설정한다.
# 2. 모든 합집합 연산을 처리할 때까지 1의 과정을 반복한다.

# 서로소 집합 자료구조 : 기본적인 구현 방법
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
    # 루트 노드를 찾을 때까지 재귀 호출
    if parent[x] != x:
        return find_parent(parent, parent[x])

# 두 원소가 속한 집합을 합치기


def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a < b:
        parent[b] = a
    else:
        parent[a] = b


# 노드의 개수와 간선의 개수 입력받기
v, e = map(int, input().split())
parent = [0] * (v+1)                # 부모 테이블 초기화

# 부모 테이블 상에서 부모를 자기 자신으로 초기화
for i in range(1, v+1):
    parent[i] = i

# Union 연산을 각각 수행
for i in range(e):
    a, b = map(int, input(). split())
    union_parent(parent, a, b)

# 각 원소가 속한 집합 출력하기
print("각 원소가 속한 집합 : ", end='')
for i in range(1, v+1):
    print(find_parent(parent, i), end=' ')

print()

# 부모 테이블 내용 출력하기
print('부모 테이블 : ', end='')
for i in range(1, v+1):
    print(parent[i], end=' ')
