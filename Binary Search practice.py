# 이진 탐색 복습

from bisect import bisect_left, bisect_right


def binary_search(start, end, target, array):
    if start > end:
        return None

    mid = (start + end) // 2

    if array[mid] == target:
        return mid
    elif array[mid] > target:
        return binary_search(start, mid-1, target, array)
    elif array[mid] < target:
        return binary_search(mid+1, end, target, array)


# n : 원소의 개수, target : 찾고자 하는 값
n, target = map(int, input().split())
array = list(map(int, input().split()))

result = binary_search(0, n-1, target, array)

if result == None:
    print("cannot find")
else:
    print(result + 1)


# 이진 탐색 라이브러리

arr = [1, 2, 4, 4, 8]
x = 4

print(bisect_right(arr, x))
print(bisect_left(arr, x))


# 특정 원소의 개수 구하기

array = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 8, 9]


def counting(array, target):
    return bisect_right(array, target) - bisect_left(array, target)


result1 = counting(array, 3)
result2 = counting(array, 4)

print(result1, result2)


# 떡볶이 떡 만들기 문제
n, m = map(int, input().split())    # n : 떡의 개수, m : 요청한 떡의 길이
tteok = list(map(int, input().split()))

tteok.sort()


def tteokFunc(array, start, end, H):
    if start > end:
        return None

    middle = (start + end) // 2

    sum = 0

    for i in range(len(array)):
        if array[i] > middle:
            sum = sum + (array[i] - middle)

    if sum == H:
        return middle
    elif sum > H:
        return tteokFunc(array, start+1, end, H)
    elif sum < H:
        return tteokFunc(array, start, end-1, H)


result = tteokFunc(tteok, 0, tteok[n-1], m)

if result == None:
    print("None")
else:
    print(result)
