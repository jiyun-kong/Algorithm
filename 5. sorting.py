# 정렬 알고리즘
# 정렬 (Sorting)이란 데이터를 특정한 기준에 따라 순서대로 나열하는 것을 말한다.
# 일반적으로 문제 상황에 따라서 적절한 정렬 알고리즘이 공식처럼 사용된다.

# 선택 정렬 : 처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복한다.
# 구현

import time
from random import randint
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(array)-1):
    for j in range(i+1, len(array)):
        if array[i] > array[j]:
            array[i], array[j] = array[j], array[i]

print(array)

# 선택 정렬의 시간 복잡도 : 선택 정렬은 N번 만큼 작은 수를 찾아서 맨 앞으로 보내야 한다.
# 구현 방식에 따라서 사소한 오차는 있을 수 있지만, 전체 횟수는 다음과 같다.
# N + N-1 + N-2 + N-3 + N-4 ... 2
# O(N^2)


# 삽입 정렬 : 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입한다.
# 선택 정렬에 비해 구현 난이도가 높은 편이지만, 일반적으로 더 효율적으로 동작한다.
# 구현

array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1):
        if array[j] < array[j-1]:
            array[j-1], array[j] = array[j], array[j-1]
        else:
            break
print(array)

# 삽입 정렬의 시간 복잡도 : 반복문이 두 번 중첩되어 사용되므로 O(n^2)
# 삽입 정렬은 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작한다.
# 최선의 경우 O(N)의 시간 복잡도를 가진다. 이미 정렬되어 있는 상태에서 다시 삽입 정렬을 수행하면 어떻게 될까?
# 0 1 2 3 4
# 1은 0과 비교하여 자신이 더 큰 것을 알고 멈출 것이다. 2는 1과 비교하여 자신이 더 큰 것을 알고 멈출 것이다. ...


# 퀵 정렬 : 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
# 일반적인 상황에서 가장 많이 사용되는 정렬 알고리즘 중 하나이다.
# 병합 정렬과 더불어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘이다.
# 가장 기본적인 퀵 정렬은 첫번째 데이터를 기준 데이터 (pivot)로 설정한다.
# 5 7 9 0 3 1 6 2 4 8
# 5 (Pivot), 7 & 4 선택, 7과 4의 위치 바뀜
# 5 4 9 0 3 1 6 2 7 8
# 9 & 2 선택, 9와 2의 위치 바뀜
# 5 4 2 0 3 1 6 9 7 8
# 6 & 1 선택, 위치가 엇갈리는 경우 '피벗'과 '작은 데이터'의 위치를 서로 변경
# 1 4 2 0 3 5 6 9 7 8
# 이제 5의 왼쪽에 있는 데이터는 모두 5보다 작고, 오른쪽에 있는 데이터는 모두 5보다 크다. 이럴게 피벗을 기준으로 데이터 묶음을 나누는 작업을 분할(Divide)라고 한다.
# 왼쪽에 있는 데이터에 대해서 정렬 수행
# 1 4 2 0 3
# 4 & 0
# 1 0 2 4 3
# 6 9 7 8
# 퀵 소트만 따로 추가 공부하기

# 이상적인 경우 분할이 절반씩 일어난다면 전체 연산 횟수로 O(NlogN)을 기대할 수 있다.
# 너비 X 높이 = N X LogN = NLogN

# 최악의 경우 O(N^2)의 시간 복잡도를 가진다. 이미 정렬된 배열에 대해서 퀵 정렬을 수행하면 그러한 결과가 나타난다.
# 구현

array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]


def quick_sort(array, start, end):
    if start >= end:
        return
    pivot = start
    left = start + 1
    right = end

    while(left <= right):
        while (left <= end and array[left] <= array[pivot]):
            left += 1
        while(right > start and array[right] >= array[pivot]):
            right -= 1
        if(left > right):
            array[right], array[pivot] = array[pivot], array[right]
        else:
            array[left], array[right] = array[right], array[left]

    quick_sort(array, start, right - 1)
    quick_sort(array, right+1, end)


quick_sort(array, 0, len(array) - 1)
print(array)


# 퀵 정렬 - 파이썬의 장점을 살린 방식

array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]


def quick_sort(array):
    if len(array) <= 1:
        return array
    pivot = array[0]
    tail = array[1:]

    left_side = [x for x in tail if x <= pivot]
    right_side = [x for x in tail if x > pivot]

    return quick_sort(left_side) + [pivot] + quick_sort(right_side)


print(quick_sort(array))


# 계수 정렬 : 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠르게 동작하는 정렬 알고리즘이다.
# 계수 정렬은 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용 가능하다.
# 데이터의 개수가 N, 데이터(양수) 중 최대값이 K일 때 최악의 경우에도 수행시간 O(N+K)를 보장한다.

# 가장 작은 데이터부터 가장 큰 데이터까지의 범위가 모두 담길 수 있도록 리스트를 생성한다.
# 7 5 9 0 3 1 6 2 9 1 4 8 0 5 2
# [2, 2, 2, 1, 1, 2, 1, 1, 1, 2]
# 리스트의 텃번째 데이터부터 하나씩 그 값만큼 반복하여 인덱스를 출력한다.
# 0 0 1 1 2 2 3 4 5 5 6 7 8 9 9

# 구현

array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
count = [0]*(max(array)+1)

for i in range(len(array)):
    count[array[i]] += 1
for i in range(len(count)):
    for j in range(count[i]):
        print(i, end=' ')


# 계수 정렬의 복잡도 분석
# 계수 정렬의 시간 복잡도와 공간 복잡도는 모두 O(N+K)이다.
# 계수 정렬은 때에 따라서 심각한 비효율성을 초래할 수도 있다. : 데이터가 0과 999999으로 단 2개가 존재하는 경우
# 계수 정렬은 동일한 값을 가지는 데이터가 여러 개 등장할 때 효과적으로 사용할 수 있다. : 성적의 경우 100점을 맞은 학생이 여러 명일 수 있기 때문에 계수 정렬이 효과적이다.

# 대부분의 프로그래밍 언어에서 지원하는 표준 정렬 라이브러리는 최악의 경우에도 O(NlogN)을 보장하도록 설계되어 있다.

# 선택 정렬과 기본 정렬 라이브러리 수행 시간 비교


# 배열에 10000개의 정수 삽입
array = []
for _ in range(10000):
    # 1부터 100 사이의 랜덤한 정수
    array.append(randint(1, 100))

# 선택 정렬 프로그램 성능 측정
start_time = time.time()

# 선택 정렬 프로그램 소스코드
for i in range(len(array)):
    min_index = i   # 가장 작은 원소의 인덱스
    for j in range(i+1, len(array)):
        if array[min_index] > array[j]:
            min_index = j
    array[i], array[min_index] = array[min_index], array[i]

# 측정 종료
end_time = time.time()

# 수행 시간 출력
print("선택 정렬 성능 특정 : ", end_time - start_time)

# 배열을 다시 무작위 데이터로 초기화
array = []
for _ in range(10000):
    array.append(randint(1, 100))

start_time = time.time()
array.sort()

end_time = time.time()
print("기본 정렬 라이브러리 성능 측정 : ", end_time - start_time)


# <문제> 두 배열의 원소 교체
# 동빈이는 두 개의 배열 A와 B를 가지고 있습니다. 두 배열은 N개의 원소로 구성되어 있으며, 배열의 원소는 모두 자연수입니다.
# 동빈이는 최대 K번의 바꿔치기 연산을 수행할 수 있는데, 바꿔치기 연산이란 배열 A에 있는 원소 하나와 배열 B에 있는 원소 하나를 골라서 두 원소를 서로 바꾸는 것을 말합니다.
# 동빈이의 최종 목표는 배열 A의 모든 원소의 합이 최대가 되도록 하는 것이며, 여러분은 동빈이를 도와야 합니다.
# N, K 그리고 배열 A와 B의 정보가 주어졌을 때, 최대 K번의 바꿔치기 연산을 수행하여 만들 수 있는 배열 A의 모든 원소의 합의 최댓값을 출력하는 프로그램을 작성하세요.

N, K = map(int, input().split())
sum = 0

A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort(reverse=True)

for k in range(K):
    if A[k] < B[k]:
        A[k], B[k] = B[k], A[k]

for a in A:
    sum += a

print(sum)


# 핵심 아이디어 : 매번 배열 A에서 가장 작은 우너소를 골라서, 배열 B에서 가장 큰 원소와 교체한다.
# 가장 먼저 배열 A와 B가 주어지면 A에 대해서 오름차순, B에 대해서 내림차순 정렬한다.
# 이후에 두 배열의 원소를 첫번째 인덱스부터 차례로 확인하면서 A의 원소가 B의 원소보다 작을 때에만 교체를 수행한다.
# 이 문제에서는 두 배열의 원소가 최대 100000개까지 입력될 수 있으므로 최악의 경우 O(NlogN)을 보장하는 정렬 알고리즘을 이용해야 한다.
