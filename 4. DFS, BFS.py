# -- coding: utf-8 --

# 이 문제는 왜 BFS인지, 왜 DFS인지 생각하기.

# 그래프 탐색 알고리즘
# 탐색 (Search)이란 많은 양의 데이터 중에서 원하는 데이터를 찾는 과정을 말한다.
# 대표적인 그래프 탐색 알고리즘으로는 DFS와 BFS가 있다.
# DFS/BFS는 코딩 테스트에서 매우 자주 등장하는 유형이므로 반드시 숙지해야 한다.

# 스택 자료구조
# 먼저 들어 온 데이터가 나중에 나가는 형실 (선입선출)의 자료구조
# 입구와 출구가 동일한 형태로 스택을 시각화할 수 있다.

# 스택 구현 예제
from collections import deque
stack = []
stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1])  # 최상단 원소부터 출력 (가장 나중에 들어온 것부터 출력)
print(stack)    # 최하단 원소부터 출력 (가장 먼저 들어온 것부터 출력)


# 큐 자료구조
# 먼저 들어 온 데이터가 먼저 나가는 형식 (선입선출)의 자료구조
# 큐는 입구와 출구가 모두 뚫려 있는 터널과 같은 형태로 시각화 할 수 있다.

# 큐 구현 예제

queue = deque()     # 큐 구현을 위해 deque (덱) 라이브러리 사용

queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()

print(queue)   # 먼저 들어온 순서대로 출력
queue.reverse()     # 역순으로 바꾸기
print(queue)    # 나중에 들어온 원소부터 출력


# 재귀 함수
# 자기 자신을 다시 호출하는 함수
# 단순한 형태의 재귀 함수 예제
# '재귀 함수를 호출합니다.'라는 문자열을 무한히 출력한다.
# 어느 정도 출력하다가 최대 재귀 깊이 초과 메시지가 출력된다.
# DFS 구현 시 필요한 문법

# 재귀 함수의 종료 조건 : 재귀 함수를 문제 풀이에서 사용할 때에는 재귀 함수의 종료 조건을 반드시 명시해야 한다.
# 종료 조건을 명시하지 않으면 함수가 무한히 호출될 수 있다.
def recursive_function(i):
    # 100번째 호출을 했을 때 종료되도록 종료 조건 명시
    if i == 100:
        return
    print(i, '번째 재귀함수에서', i+1, '번째 재귀함수를 호출합니다.')
    recursive_function(i+1)
    print(i, '번째 재귀함수를 종료합니다.')


recursive_function(1)

# 팩토리얼 구현 예제
# 수학적으로 0!과 1!의 값은 1이다.


def factorial(n):
    if n <= 1:
        return 1
    else:
        return n*factorial(n-1)


result = factorial(5)
print(result)


# 최대공약수 계산 (유클리드 호제법) 예제
# 두 개의 자연수에 대한 최대공약수를 구하는 대표적인 알고리즘으로 유클리드 호제법이 있다.
num1 = int(input())
num2 = int(input())


def euclid(num1, num2):

    if num1 % num2 == 0:
        return num2
    else:
        mod = num1 % num2
        return euclid(num2, mod)


if(num1 > num2):
    result = euclid(num1, num2)
else:
    result = euclid(num2, num1)

print(result)


# 이렇게 짜면 더 큰 값, 더 작은 값 상관 없다.

def gcd(a, b):
    if a % b == 0:
        return b
    else:
        return gcd(b, a % b)


print(gcd(192, 162))

# 재귀 함수 사용의 유의 사항
# 재귀 함수를 잘 활용하면 복잡한 알고리즘을 간결하게 작성할 수 있다.
# 단, 오히려 다른 사람이 이해하기 어려운 형태의 코드가 될 수도 있으므로 신중하게 사용해야 한다.
# 모든 재귀 함수는 반복문을 이용하여 동일한 기능을 구현할 수 있다.
# 재귀 함수가 반복문보다 유리한 경우도 있고 불리한 경우도 있다.
# 컴퓨터가 함수를 연속적으로 호출하면 컴퓨터 메모리 내부의 스택 프레임에 쌓인다.
# 그래서 스택을 사용해야 할 때 구현상 스택 라이브러리 대신에 재귀 함수를 이용하는 경우가 많다.


# DFS (Depth-First Search)
# DFS는 깊이 우선 탐색이라고도 부르며 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘이다.
# DFS는 스택 자료구조 (혹은 재귀 함수)를 이용하며, 구체적인 동작 과정은 다음과 같다.
# 1. 탐색 시작 노드를 스택에 삽입하고 방문 처리를 한다.
# 2. 스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면 그 노드를 스택에 넣고 방문 처리한다. 방문하지 않은 인접 노드가 없다면 스택에서 최상단 노드를 꺼낸다.
# 3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.

# DFS 소스코드 예제

# DFS 메서드 정의
def dfs(graph, v, visited):

    # 현재 노드를 방문 처리
    visited[v] = True
    print(v, end='')

    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)


# 각 노드가 연결된 정보를 표현 (2차원 리스트)
graph = [
    [],             # 0번 노드와 연결된 노드는 없다
    [2, 3, 8],        # 1번 노드와 연결된 노드는 2,3,8이다
    [1, 7],          # 2번 노드와 연결된 노드는 1,7이다
    [1, 4, 5],        # 3번 노드와 연결된 노드는 1,4,5이다
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가 방문된 정보를 표현 (1차원 리스트)
visited = [False]*9     # 0번 노드는 사용 안하게 일부러 9개의 원소로 만들어줌

# 정의된 DFS 함수 호출
dfs(graph, 1, visited)


# BFS (Breadth-First Search)
# BFS는 너비 우선 탐색이라고도 부르며, 그래프에서 가까운 노드부터 우선적으로 탐색하는 알고리즘이다.
# 실제 기업 코딩 케스트에 자주 등장하는 문제이며, 최단경로를 파악하는 문제에서 자주 등장한다.
# BFS는 큐 자료구조를 이용하며, 구체적인 동작 과정은 다음과 같다.
# 1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다.
# 2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문처리한다.
# 3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.


# BFS 메서드 정의

def bfs(graph, start, visited):
    # 큐 (Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력하기
        v = queue.popleft()
        print(v, end=' ')

        # 아직 방문하지 않은 인접한 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True


# 각 노드가 연결된 정보를 표현 (2차원 리스트)
graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가 방문된 정보를 표현 (1차원 리스트)
visited = [False]*9

# 정의된 BFS 함수 호출
bfs(graph, 1, visited)


# <문제> 음료수 얼려 먹기
# N x M 크기의 얼음 틀이 있다. 구멍이 뚫려 있는 부분은 0, 칸막이가 존재하는 부분은 1로 표시된다. 구멍이 꿇려 있는 부분끼리 상, 하, 좌, 우로 붙어 있는 경우 서로 연결되어 있는 것으로 간주한다.
# 이때 얼음 틀의 모양이 주어졌을 때 생성되는 총 아이스크림의 개수를 구하는 프로그램을 작성하라.
# 다음의 4 x 5 얼음 틀 예씨에서는 아이스크림이 총 3개 생성된다.
# 연결 요소 찾기 문제이다. (Connected Component)

# 해결이 쉽지 않다...
# 이 문제는 DFS 혹은 BFS로 해결할 수 있다. 일단 앞에서 배운 대로 얼음을 얼릴 수 있는 공간이 상, 하, 좌, 우로 연결되어 있다고 표현할 수 있으므로 그래프 형태로 모델링 할 수 있다.
# DFS를 활용하는 알고리즘은 다음과 같다
# 1. 특정한 지점의 주변 상, 하, 좌, 우를 살펴본 뒤에 주변 지점 중에서 값이 0이면서 아직 방문하지 않은 지점이 있다면 해당 지점을 방문한다.
# 2. 방문한 지점에서 다시 상, 하, 좌, 우를 살펴보면서 방문을 진행하는 과정을 반복하면, 연결된 모든 지점을 방문할 ㅅ ㅜ있다.
# 3. 모든 노드에 대하여 1~2번의 과정을 반복하며, 방문하지 않은 지점의 수를 카운트한다.

# DFS로 특정 노드를 방문하고 연결된 모든 노드들도 방문
n, m = map(int, input().split())

graph = []
for i in range(n):
    graph.append(list(map(int, input())))


def dfs(x, y):
    # 주어진 범위를 벗어나는 경우에는 즉시 종료
    if x <= -1 or x >= n or y <= -1 or y >= m:
        return False

    # 현재 노드를 아직 방문하지 않았다면
    if graph[x][y] == 0:
        graph[x][y] = 1
        dfs(x-1, y)
        dfs(x, y-1)
        dfs(x+1, y)
        dfs(x, y+1)
        return True
    return False


# # 모든 노드 (위치)에 대해여 음료수 채우기
result = 0
for i in range(n):
    for j in range(m):
        # 현재 위치에서 DFS 수행
        if dfs(i, j) == True:
            result += 1

print(result)


n, m = map(int, input().split())

graph = []
for num in range(n):
    graph.append(list(map(int, input())))


def ice_making(i, j):
    if i < 0 or i > (n-1) or j < 0 or j > (m-1):
        return False

    if graph[i][j] == 1:
        return False
    elif graph[i][j] == 0:
        graph[i][j] = 1
        ice_making(i-1, j)
        ice_making(i+1, j)
        ice_making(i, j-1)
        ice_making(i, j+1)
        return True


ice = 0

for i in range(n):
    for j in range(m):
        if ice_making(i, j):
            ice += 1

print(ice)


# <문제>
# 동빈이는 N x M 크기의 직사각형 형태의 미로에 갇혔다. 미로에는 여러 마리의 괴물이 있어 이를 피해 탈출해야 한다.
# 동빈이의 위치는 (1,1)이며 미로의 출구는 (N,M)의 위치에 존재하며 한 번에 한 칸씩 이동할 수 있다.
# 이때 괴물이 있는 부분은 0으로, 괴물이 없는 부분은 1로 표시되어 있다. 미로는 반드시 탈출할 수 있는 형태로 제시된다.
# 이때 동빈이가 탈출하기 위해 움직여야 하는 최소 칸의 개수를 구하시오. 칸을 셀 때는 시작 칸과 마지막 칸을 모두 포함해서 계산한다.


n, m = map(int, input().split())
graph = []

for num in range(n):
    graph.append(list(map(int, input())))

step = 0


def maze(i, j):
    global step

    if i < 0 or i > n-1 or j < 0 or j > m-1:
        return False

    if graph[i][j] == 1:
        graph[i][j] = 0
        maze(i+1, j)
        maze(i, j+1)

        step += 1
    elif graph[i][j] == 0:
        return False


maze(0, 0)

print(step)


# BFS는 시작 지점에서 가까운 노드부터 차례대로 그래프의 모든 노드를 탐색한다.
# 상, 하, 좌, 우로 연결된 모든 노드로의 거리가 1로 동일하다.
# 따라서 (1,1) 지점부터 BFS를 수행하여 모든 노드의 최단거리 값을 기록하면 해결할 수 있다.
# 대박이다...ㅋ
# 처음에 (1,1)의 위치에서 시작한다.
# 좌표에서 상, 하, 좌, 우로 탐색을 진행하면 바로 옆 노드인 (1,2) 위치의 노드를 방문하게 되고 새롭게 방문하는 (1,2) 노드의 값을 2로 바꾼다.
# 계속해서 BFS를 수행하면서 최단 경로의 값들이 1씩 증가하는 형태로 변경한다.


def bfs(x, y):
    queue = deque()
    queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]

            # 미로 찾기 공간을 벗어난 경우 무시
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue

            # 벽인 경우 무시
            if graph[nx][ny] == 0:
                continue

            # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
            if graph[nx][ny] == 1:
                graph[nx][ny] = graph[x][y] + 1
                queue.append((nx, ny))

    # 가장 오른쪽 아래까지의 최단 거리 반환
    return graph[n-1][m-1]


n, m = map(int, input().split())
graph = []

for i in range(n):
    graph.append(list(map(int, input())))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

print(bfs(0, 0))
