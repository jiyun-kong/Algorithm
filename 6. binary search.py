from bisect import bisect_left, bisect_right
# 이진 탘색 알고리즘
# 순차 탐색 : 리스트 안에 있는 특정한 데이터를 찾기 위해 앞에서부터 데이터를 하나씩 확인하는 방법
# 이진 탐색 : 정렬되어 있는 리스트에서 탐색 범위를 절반씩 좁혀가면 데이터를 탐색하는 방법 : 시작점, 끝점, 중간점을 이용하여 탐색 범위를 설정한다.

# 단계마다 탐색 범위를 2로 나누는 것과 동일하므로 연산 횟수는 log2N에 비례한다.
# 이진 탐색은 탐색 범위를 절반씩 줄이며, 시간 복잡도는 O(logN)을 보장한다.

# 구현


def binary_search(array, target, start, end):
    if start > end:
        return None

    mid = (start + end) // 2

    if target == array[mid]:
        return mid
    elif target > array[mid]:
        return binary_search(array, target, mid+1, end)
    elif target < array[mid]:
        return binary_search(array, target, start, mid-1)


# n : 원소의 개수, target : 찾고자 하는 값
n, target = map(int, input().split())

# 전체 원소 입력 받기
array = list(map(int, input().split()))

result = binary_search(array, target, 0, n-1)

if result == None:
    print("Cannot Find")
else:
    print(result+1)


# 파이썬 이진 탐색 라이브러리
# bisect_left(a, x) : 정렬된 순서를 유지하면서 배열 a에 x를 삽입할 가장 왼쪽 인덱스를 반환
# bisect_right(a, x) : 정렬된 순서를 유지하면서 배열 a에 x를 삽입할 가장 오른쪽 인덱스를 반환


a = [1, 2, 4, 4, 8]
x = 4

print(bisect_right(a, x))   # 4
print(bisect_left(a, x))     # 2


# 값이 특정 범위에 속하는 데이터 개수 구하기


def count_by_range(a, left_value, right_value):
    right_index = bisect_right(a, right_value)
    left_index = bisect_left(a, left_value)

    return right_index - left_index


a = [1, 2, 3, 3, 3, 3, 4, 4, 8, 9]

print(count_by_range(a, 4, 4))        # 2

print(count_by_range(a, -1, 3))     # 6


# 파라메트릭 서치 (parametric search)
# 파라메트릭 서치 : 최적화 문제를 결정 문제 ('예' 혹은 '아니오')로 바꾸어 해결하는 기법
# 예시 : 특정한 조건을 만족하는 가장 알맞은 값을 빠르게 찾는 최적화 문제
# 일반적으로 코딩 테스트에서 파라매트릭 서치 문제는 이진 탐색을 이용하여 해결할 수 있다.

# <문제> 떡볶이 떡 만들기
# 오늘 동빈이는 여행 가신 부모님을 대신해서 떡집 일을 하기로 했습니다. 오늘은 떡볶이 떡을 만드는 날입니다. 동빈이네 떡볶이 떡은 재밌게도 떡볶이 떡의 길이가 일정하지 않습니다.
# 대신에 한 봉지 안에 들어가는 떡의 총 길이는 절단기로 잘라서 맞춰줍니다.
# 절단기에 높이 (H)를 지정하면 줄지어진 떡을 한 번에 절단합니다. 높이가 H보다 긴 떡은 H 위의 부분이 잘릴 것이고, 낮은 떡은 잘리지 않습니다.
# 예를 들어 높이가 19, 14, 10, 17cm인 떡이 나란히 있고 절단기 높이를 15cm로 지정하면 자른 뒤 떡의 높이는 15, 14, 10, 15cm가 될 것입니다. 잘린 떡의 길이는 차례대로 4, 0, 0, 2cm입니다.
# 손님은 6cm 만큼의 길이를 가져갑니다.
# 손님이 왔을 때 요청한 통 길이가 M일 때 적어도 M만큼의 떡을 얻기 위해 절단기에 설정할 수 있는높이의 최댓값을 구하는 프로그램을 작성하세요

# 떡의 개수 N, 요청한 떡의 길이 M
# 간격이 길면 어카냐? 흠... 모르겠음

N, M = map(int, input().split())
tteok = list(map(int, input().split()))


def cut_tteok(tteok, start_idx, end_idx, target):
    sum = 0

    if start_idx > end_idx:
        return None

    start = tteok[start_idx]
    end = tteok[end_idx]

    mid = (start + end) // 2

    for t in range(start_idx, end_idx+1):
        if tteok[t] > mid:
            sum += tteok[t] - mid

    if sum == target:
        return mid
    elif sum < target:
        return cut_tteok(tteok, start-1, end, target)
    elif sum > target:
        return cut_tteok(tteok, start+1, end, target)


tteok.sort()
H = cut_tteok(tteok, 0, N-1, M)

if H == None:
    print("impossible")
else:
    print(H)


# 적절한 높이를 찾을 때까지 이진 탐색을 수행하여 높이 H를 반복해서 조정하면 된다.
# '현재 이 높이로 자르면 조건을 만족할 수 있는가?'를 확인한 뒤에 조건의 만족 여부 ('예' 혹은 '아니오')에 따라서 탐색 범위를 좁혀서 해결할 수 있다.
# 절단기의 높이는 0 ~ 10억 정수 중 하나이다. : 이렇게 큰 탐색 범위를 보면 가장 먼저 이진 탐색을 떠올려야 한다.
# 중간점의 값은 시간이 지날수록 '최적화된 값'이 되기 때문에, 과정을 반복하면서 얻을 수 있는 떡의 길이 합이 필요한 떡의 길이보다 크거나 같을 때마다 중간점의 값을 기록하면 된다.

N, M = map(int, input().split())
tteok = list(map(int, input().split()))

start = 0
end = max(tteok)

result = 0
while (start <= end):
    total = 0
    mid = (start + end) // 2

    for x in tteok:
        if x > mid:
            total += x - mid

    if total < M:
        end = mid - 1
    else:
        result = mid
        start = mid + 1

print(result)


# < 문제 > : 정렬된 배열에서 특정 수의 개수 구하기
# N개의 원소를 포함하고 있는 수열이 오름차순ㅇ로 정렬되어 있다. 이때 이 수열에서 x가 등장하는 횟수를 계산허시오. 예를 들어 수열 {1, 1, 2, 2, 2, 2, 3}이 있을 때
# x = 2이면, 현재 수열에서 값이 2인 원소가 4개이므로 4를 출력한다.
# 단 이 문제는 시간 복잡도 O(logN)으로 알고리즘을 설계하지 않으면 시간 초과 판정을 받는다.

N, x = map(int, input().split())
arr = list(map(int, input().split()))


def targeting(arr, left_value, right_value):
    right_index = bisect_right(arr, right_value)
    left_index = bisect_left(arr, left_value)

    return right_index - left_index


result = targeting(arr, x, x)

if result == 0:
    print("-1")
else:
    print(result)


# 시간 복잡도 O(logN)으로 동작하는 알고리즘을 요구하고 있다.
# 일반적인 선형 탐색으로는 시간 초과 판정을 받는다. 하지만 데이터가 정렬되어 있기 때문에 이진 탐색을 수행할 수 있다.
# 특정 값이 등장하는 첫 번째 위치와 마지막 위치를 찾아 위치 차이를 계산해 문제를 해결할 수 있다.

# 표준 라이브러리를 사용하지 않았을 때... 다시 생각해보기
N, x = map(int, input().split())
arr = list(map(int, input().split()))

start = 0
end = N-1


def startIdx(arr, start, end):
    mid = (start + end) // 2

    if x == arr[mid]:
        return
