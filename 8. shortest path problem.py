# 최단 경로 알고리즘 : 가장 짧은 경로를 찾는 알고리즘
# 다양한 문제 상황
# 1) 한 지점에서 다른 한 지점까지의 최단 경로
# 2) 한 지점에서 다른 모든 지점까지의 최단 경로
# 3) 모든 지점에서 다른 모든 지점까지의 최단 경로

# 각 지점은 그래프에서 노드로 표현
# 지점 간 연결된 도로는 그래프에서 간선으로 표현

# 다익스트라 최단 경로 알고리즘 개요
# 특정한 노드에서 출발하여 다른 모든 노드로 가는 최단 경로를 계산한다.
# 다익스트라 최단 경로 알고리즘은 음의 간선이 없을 때 정상적으로 동작한다. : 현실 세계의 도로 (간선)은 음의 간선으로 표현되지 않는다.
# 다익스트라 최단 경로 알고리즘은 그리디 알고리즘으로 분류된다. : 매 상황에서 가장 비용이 적은 노드를 선택해 임의의 과정을 반복한다.

# 알고리즘의 동작 과정
# 1) 출발 노드를 설정한다.
# 2) 최단 거리 테이블을 초기화한다.
# 3) 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택한다.
# 4) 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블을 갱신한다.
# 5) 위 과정에서 3번과 4번을 반복한다.

# 다익스트라 알고리즘의 특징
# 그리디 알고리즘 : 매 상황에서 방문하지 않은 가장 비용이 적은 노드를 선택해 임의의 과정을 반복한다.
# 단계를 거치며 한 번 처리된 노드의 최단 거리는 고정되어 더이상 바뀌지 않는다. : 한 단계당 하나의 노드에 대한 최단거리를 확실히 찾는 것으로 이해할 수 있다.
# 다익스트라 알고리즘을 수행한 뒤에 테이블에 각 노드까지의 최단 거리 정보가 저장된다. : 완벽한 형태의 최단 경로를 구하려면 소스코드에 추가적인 기능을 더 넣어야 한다.

# 다익스트라 알고리즘 : 간단한 구현 방법
# 단계마다 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택하기 위해 매 단계마다 1차원 테이블의 모든 원소를 확인 (순차 탐색)한다.
import heapq
import sys
input = sys.stdin.readline
INF = int(1e9)      # 무한을 의미하는 값을 10억을 설정

# 노드의 개수, 간선의 개수를 입력받기
n, m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]
# 방문한 적이 있는지 체크하는 목적의 리스트를 만들기
visited = [False] * (n+1)
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())

    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))

# 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환


def get_smallest_node():
    min_value = INF
    index = 0

    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index


def dijkstra(start):
    # 시작 노드에 대해서 초기화
    distance[start] = 0
    visited[start] = True
    for j in graph[start]:
        distance[j[0]] = j[1]

    # 시작 노드를 제외한 전체 (n-1)개의 노드에 대해 반복
    for i in range(n-1):
        # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리
        now = get_smallest_node()
        visited[now] = True

        # 현재 노드와 연결된 다른 노드를 확인
        for j in graph[now]:
            cost = distance[now] + j[1]

            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[j[0]]:
                distance[j[0]] = cost


dijkstra(start)

for i in range(1, n+1):
    # 도달할 수 없는 경우, 무한이라고 출력
    if distance[i] == INF:
        print("INFINITY")

    # 도달할 수 있는 경우 거리를 출력
    else:
        print(distance[i])


# 총 O(V)번에 걸쳐서 최단 거리가 가장 짧은 노드를 매번 선형 탐색해야 한다.
# 따라서 전체 시간 복잡도는 O(V^2)이다.
# 일반적으로 코딩 테스트의 최단 경로 문제에서 전체 노드의 개수가 5000개 이하라면 이 코드로 문제를 해결할 수 있다. 하지만 노드의 개수가 10000개를 넘어가는 문제라면 어떻게 해야할까?
# (참고로 파이썬의 연산 속도는 1초에 약 2000만번)


# 우선순위 큐 (Priority Queue) : 우선 순위가 가장 높은 데이터를 가장 먼저 삭제하는 자료구조
# 예를 들어 여러 개의 물건 데이터를 자료구조에 넣었다가 가치가 높은 물건 데이터부터 꺼내서 확인해야 하는 경우에 우선순위 큐를 이용할 수 있다.
# 파이썬, c++, Java를 포함한 대부분의 프로그래밍 언어에서 표준 라이브러리 형태로 지원한다.

# 자료구조 별 추출되는 데이터
# 스택 : 가장 나중에 삽입된 데이터
# 큐 : 가장 먼저 삽입된 데이터
# 우선순위 큐 : 가장 우선순위가 높은 데이터

# 힙 (Heap) : 우선순위 큐를 구현하기 위해 사용하는 자료구조 중 하나이다.
# 최소 힙 (Min Heap)과 최대 힙 (Max Heap)이 있다.
# 다익스트라 최단 경로 알고리즘을 포함해 다양한 알고리즘에서 사용된다.

# 우선순위 큐 구현 방식     /   삽입 시간   / 삭제 시간
# 리스트                   /   O(1)       / O(N)
# 힙 (Heap)                /   O(logN)    / O(logN)


# 힙 라이브러리 사용 예제 : 최소 힙 - 제공됨

# 오름차순 힙 정렬 (Heap sort)

def heapsort(iterable):
    h = []
    result = []

    # 모든 원소를 차례대로 힙에 삽입
    for value in iterable:
        heapq.heappush(h, value)

    # 힙에 삽입된 모든 원소를 차례때로 꺼내어 담기
    for i in range(len(h)):
        result.append(heapq.heappop(h))
    return result


result = heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
print(result)


# 힙 라이브러리 사용 예제 : 최대 힙 - 제공되지 않음 : 부호를 넣어서 삽입했다가 부호를 빼고 꺼내면 된다.
def heapsort(iterable):
    h = []
    result = []

    # 모든 원소를 차례대로 힙에 삽입
    for value in iterable:
        heapq.heappush(h, -value)

    # 힙에 삽입된 모든 원소를 차례때로 꺼내어 담기
    for i in range(len(h)):
        result.append(-heapq.heappop(h))
    return result


result = heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
print(result)


# 다익스트라 알고리즘 : 개선된 구현 방법
# 단계마다 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택하기 위해 힙 (Heap) 자료구조를 이용한다.
# 다익스트라 알고리즘이 동작하는 기본 원리는 동일하다.
# 현재 가장 가까운 노드를 저장해 놓기 위해서 힙 자료구조를 추가적으로 이용한다는 점이 다르다.
# 현재의 최단 거리가 가장 짧은 노드를 선택해야 하므로 최소 힙을 사용한다.

# 다익스트라 알고리즘 : 개선된 구현 방법 (Python)
input = sys.stdin.readline
INF = int(1e9)  # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수를 입력받기
n, m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))


def dijkstra(start):
    q = []
    # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여 큐에 삽입
    heapq.heappush(q, (0, start))
    distance[start] = 0

    while q:    # 큐가 비어있지 않다면
        # 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
        dist, now = heapq.heappop(q)
        # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
        if distance[now] < dist:
            continue
        # 현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
            cost = dist + i[1]
            # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))


# 다익스트라 알고리즘을 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1, n+1):
    # 도달할 수 없는 경우, 무한이라고 출력
    if distance[i] == INF:
        print("Infinity")
    # 도달할 수 있는 경우 거리를 출력
    else:
        print(distance[i])

# 다익스트라 알고리즘 : 개선된 구현 방법 성능 분석
# 힙 자료구조를 이용하는 다익스트라 알고리즘의 시간 복잡도는 O(ElogV)이다.
# 노드를 하나씩 꺼내 검사하는 반복문 (while문)은 노드의 개수 V 이상의 횟수로는 처리되지 않는다. 
# : 결과적으로 현재 우선순위 큐에서 꺼낸 노드와 연결된 다른 노드들을 확인하는 총횟수는 최대 간선의 개수 (E)만큼 연산이 수행될 수 있다.

# 직관적으로 전체 과정은 E개의 원소를 우선순위 큐에 넣었다가 모두 빼내는 연산과 매우 유사하다.
# 시간 복잡도를 O(ElogE)로 판단할 수 있다.
# 중복 간선을 포함하지 않는 경우에 이를 O(ElogV)로 정리할 수 있다.


# 플로이드 워셜 알고리즘
# 모든 노드에서 다른 노드까지의 최단 경로를 모두 계산한다.
# 플로이드 워셜 (Floyd-Warshall) 알고리즘은 다익스트라 알고리즘과 마찬가지로 단계별로 거쳐 가는 노드를 기준으로 알고리즘을 수행한다.
# 다만 매 단계마다 방문하지 않은 노드 중에 최단 거리를 갖는 노드를 찾는 과정이 필요하지 않다.
# 플로이드 워셜은 2차원 테이블에 최단 거리 정보를 저장한다.
# 플로이드 워셜 알고리즘은 다이나믹 프로그래밍 유형에 속한다. -> 점화식을 이용하여 삼중 반복문으로 2차원 테이블을 갱신!
# 노드의 개수가 적을 때 효과적이다.
# (참고 : 노드와 간선의 개수가 많을 때는 다익스트라 알고리즘을 사용하는 것이 효율적)

# 각 단계마다 특정한 노드 k를 거쳐 가는 경우를 확인한다. : a에서 b로 가는 최단 거리보다 a에서 k를 거쳐 b로 가는 거리가 더 짧은지 검사한다.
# D_ab = min(D_ab, D_ak + D_kb)
# 예를 들어서 1번 노드를 거쳐 가는 경우를 고려해본다.
# 이때, 자기 자신에서 자기 자신으로 가는 경우 / 1번 노드에서 시작하는 경우 (1번 행) / 1번 노드로 끝나는 경우 (1번 열) 는 제외한다.

# 플로이드 워셜 알고리즘
INF = int(1e9)

# 노드의 개수 및 간선의 개수 입력받기
n = int(input())
m = int(input())

# 2차원 리스트 (그래프 표현)를 만들고, 무한으로 초기화
graph = [[INF] * (n+1) for _ in range(n+1)]

# 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
for a in range(1, n+1):
    for b in range(1, n+1):
        if a == b :
            graph[a][b] = 0
            
# 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
for _ in range(m):
    # A에서 B로 가는 비용은 C라고 설정
    a, b, c = map(int, input().split())
    graph[a][b] = c
    
# 점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1, n+1):
    for a in range(1, n+1):
        for b in range(1, n+1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])
            
# 수행된 결과를 출력
for a in range(1, n+1):
    for b in range(1, n+1):
        # 도달할 수 없는 경우, 무한 (Infinity)이라고 출력
        if graph[a][b] == INF:
            print("Infinity", end=" ")
        else:
            print(graph[a][b], end=" ")
        
    print()
    
# 플로이드 워셜 알고리즘 성능 분석
# 노드의 개수가 N개일 때 알고리즘 상으로 N번의 단계를 수행한다. : 각 단계마다 O(N^2)의 연산을 통해 현재 노드를 거쳐 가는 모든 경로를 고려한다.
# 따라서 플로이드 워셜 알고리즘의 총 시간 복잡도는 O(N^3)이다. 시간 복잡도가 많이 크므로 조심해서 사용해야 한다.


# < 문제 > : 전보
# 어떤 나라에는 N개의 도시가 있다. 그리고 각 도시는 보내고자 하는 메시지가 있는 경우, 다른 도시로 전보를 보내서 다른 도시로 해당 메시지를 전송할 수 있다.
# 하지만 X라는 도시에서 Y라는 도시로 전보를 보내고자 한다면, 도시 X에서 Y로 향하는 통로가 설치되어 있어야 한다. 예를 들어 X에서 Y로 향하는 통로는 있지만, Y에서 X로 향하는 통로가 없다면
# Y는 X로 메시지를 보낼 수 없다. 또한 통로를 거쳐 메시지를 보낼 때에는 일정 시간이 소요된다.
# 어느 날 C라는 도시에서 위급 상황이 발생했다. 그래서 최대한 많은 도시로 메시지를 보내고자 한다. 메시지는 도시 C에서 출발하여 각 도시 사이에 설치된 통로를 거쳐, 최대한 많이 퍼져나갈 것이다.
# 각 도시의 번호와 통로가 설치되어 있는 정보가 주어졌을 때, 도시 C에서 보낸 메시지를 받게 되는 도시의 개수는 총 몇 개이며 도시들이 모두 메시지를 받는 데까지 걸리는 시간은 얼마인지 계산하는 프로그램을 작성하시오.

import heapq
import sys
input = sys.stdin.readline
INF = int(1e9)

n, m, city = map(int, input().split())
graph = [[] for i in range(n+1)]
distance = [INF] * (n+1)

for _ in range(m):
    x, y, z = map(int, input().split())
    graph[z].append((y, z))


def dijkstra(city):
    queue = []

    heapq.heappush(queue, (0, city))    # queue에다가 시작 노드로 가기 위한 최단 거리를 0으로 설정.
    distance[city] = 0

    while queue:
        dist, now = heapq.heappop(queue)    # 가장 최단 거리가 짧은 노드를 pop 해온다.

        if distance[now] < dist:
            continue

        for i in graph[now]:
            cost = dist + i[1]

            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(queue, (cost, i[0]))


dijkstra(city)

count = 0

max_distance = 0
for d in distance:
    if d != 1e9:
        count += 1
        max_distance = max(max_distance, d)

print(count-1, max_distance)
