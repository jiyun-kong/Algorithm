# from itertools import permutations
# from itertools import combinations
# import sys
# -*- coding: utf-8 - *-

# 자료형
# 정수형

# a = 777
# print(a)

# a += 1
# print(a)

# # 실수형
# fl = 157.68
# print(fl)

# fll = 5.
# print(fll)

# a = int(1e9)
# print(a)


# # cf : 이진법을 사용하면서 나타날 수 있는 오류 -> round() 함수로 해결
# a = 0.3 + 0.9
# print(a)

# if a == 0.9:
#     print(True)
# else:
#     print(False)

# rd = 123.456
# rd = round(rd, 2)
# print(rd)

# a = 0.3 + 0.6
# print(round(a, 4))

# if round(a, 4) == 0.9:
#     print(True)
# else:
#     print(False)


# # 수 자료형의 연산
# a = 7
# b = 3

# print(a / b)
# print(a % b)
# print(a // b)
# print(a ** b)


# # 리스트 자료형
# # 리스트 초기화 : [], list()
# a = [7, 5, 7, 4, 3, 6, 5]
# print(a)
# a[4] = 100
# print(a)
# print(a[-1])
# print(a[-3])
# print(a[:2])

# # 리스트 컴프리헨션
# array = [i for i in range(10)]      # 반복문 먼저! : for i in range(10)
# print(array)

# arr = [i*2 for i in range(5)]
# print(arr)

# arr2 = [i for i in range(20) if i % 2 == 1]
# print(arr2)

# # 리스트 컴프리헨션은 특히 2차원 리스트를 초기화할 때 효과적
# n = 4
# m = 3
# array = [[0]*m for _ in range(n)]   # [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
# # arr = [[0] * m] * n   : 전체 리스트 안에 포함된 각 리스트가 모두 같은 객체로 인식됨
# # 위의 안 좋은 예시에서 arr[1][1] = 5를 하면 [[0,5,0], [0,5,0], [0,5,0], [0,5,0]] 이렇게 바뀜.

# # 파이썬에서의 언더바(_) : 반복을 수행하되 반복을 위한 변수의 값을 무시하고자 할 때 사용
# for _ in range(5):
#     print("Hello World")

# # append(), sort(), sort(reverse = True), reverse(), insert(), count(), remove()
# a = [4, 3, 2, 1]
# print(a)
# a.reverse()
# print(a)
# a.insert(2, 3)
# print(a)
# print(a.count(3))
# a.remove(1)
# print(a)

# 리스트에서 특정 값을 가지는 원소를 모두 제거하기
# a = [1, 2, 3, 4, 5, 5, 5]
# remove_set = {3, 5}     # 집합 자료형

# # remove_list에 포함되지 않은 값만을 저장
# result = [i for i in a if i not in remove_set]
# print(result)           # [1,2,4]


# # 문자열 자료형
# a = "Hello"
# # a[2] = 'J'
# print(a)


# # 튜플 자료형 : 공간 효율적
# tup = (1, 2, 3, 4, 5, 6, 7)
# print(tup[2: 4])

# 튜플을 사용하면 좋은 경우 : 서로 다른 성질의 데이터를 묶어서 관리해야 할 때
# 최단 경로 알고리즘 (비용, 노드 번호)의 형태로 튜플 자료형을 자주 사용한다.
# 데이터의 나열을 해싱 (Hashing)의 키 값으로 사용해야 할 때
# 튜플은 변경이 불가능하므로 리스트와 다르게 키 값으로 사용될 수 있다.
# 리스트보다 메모리를 효율적으로 사용해야 할 때


# 사전 자료형 : 해시 테이블을 이용하므로 데이터의 조회 및 수정에 있어서 O(1)의 시간에 처리할 수 있음
# data = dict()
# data['사과'] = "apple"
# data['바나나'] = "banana"

# print(data)

# if '사과' in data:
#     print("'사과'를 키로 가지는 데이터가 존재합니다.")

# print(data.keys())
# print(data.values())

# for key in data.keys():
#     print(data[key])

# key_list = list(data.keys())
# print(key_list)


# # 집합 자료형 : 중복 허용 안함, 순서 없음
# data = set([1, 1, 1, 2, 3, 4, 4, 4, ])
# print(data)

# data2 = {1, 2, 3, 4, 5, 5, 5, 5}
# print(data2)


# 표준 입력 방법
# input() : 한 줄의 문자열을 입력 받는 함수
# map() : 리스트의 모든 원소에 각각 특정한 함수를 적용할 때 사용
# n = int(input())
# data = list(map(int, input().split()))
# data.sort(reverse=True)
# print(data)

# a, b, c = map(int, input().split())
# print(a, b, c)

# 사용자로부터 입력을 최대한 빠르게 받아야 하는 경우 : sys 라이브러리에 정의되어 있는 sys.stdin.readline() 사용
# 입력 후 엔터가 줄 바꿈 기호로 입력되므로 rstrip() 메서드를 함께 사용

# data = sys.stdin.readline().rstrip()
# print(data)

# f-string : 문자열 앞에 f를 붙여 사용한다.
# answer = 7
# print(f"정답은 {answer}입니다.")


# 조건문
# if True or False:
#     print("Yes")

# a = 15
# if a <= 20 and a >= 10:
#     print("Yeah")

# 조건부 표현식
# score = 85
# result = "Success" if score >= 80 else "Fail"
# print(result)


# 반복문 : while문 보다는 for문을 더 많이 사용
# scores = [90, 85, 77, 65, 97]
# cheating_student_list = {2, 4}

# for i in range(5):
#     if i+1 in cheating_student_list:
#         continue
#     if scores[i] >= 80:
#         print(i+1, "번 학생은 합격입니다.")


# for dan in range(2, 10):
#     for i in range(1, 10):
#         print(dan, " x ", i, " = ", dan*i)
#     print()


# 함수: 내장 함수(파이썬이 기본적으로 제공하는 함수), 사용자 정의 함수(개발자가 직접 정의하여 사용할 수 있는 함수)
# def add(a, b):
#     return a+b


# result = add(3, 7)
# print(result)

# a = 0


# def func():
#     global a
#     a += 1
#     print(a)


# func()
# func()
# print(a)


# array = [1, 2, 3, 4, 5]

# array는 global 없이도 참조 가능, 지역 변수의 우선순위가 더 높음


# def func2():
#     array.append(6)
#     print(array)


# func2()


# def func3():
#     global array
#     array = [3, 4, 5]
#     array.append(6)
#     print(array)


# func3()


# def operator(a, b):
#     add_var = a+b
#     sub_var = a-b
#     mul_var = a*b
#     div_var = a/b
#     return add_var, sub_var, mul_var, div_var


# a, b, c, d = operator(7, 3)
# print(a, b, c, d)


# 람다 표현식: lambda parameters: return값
#print((lambda a, b: a+b)(3, 7))

# 람다 표현식 예시: 내장 함수에서 자주 사용되는 람다 함수
# array = [('홍갈동', 50), ('이순신', 32), ('아무개', 74)]


# def my_key(x):
#     return x[1]


# print(sorted(array, key=my_key))
# print(sorted(array, key=lambda x: x[1]))


# list1 = [1, 2, 3, 4, 5]
# list2 = [6, 7, 8, 9, 10]

# result = map(lambda a, b: a+b, list1, list2)
# print(list(result))

# result2 = map(lambda x: x**2, list1)
# print(list(result2))

# result3 = map(lambda x, y: x-y, list1, list2)
# print(list(result3))


# 실전에서 유용한 표준 라이브러리
# 1. 내장 함수: 기본 입출력 함수부터 정렬 함수까지 기본적인 함수들을 제공
# 2. itertools: 파이썬에서 반복되는 형태의 데이터를 처리하기 위한 유용한 기능들을 제공 - 순열과 조합 라이브러리
# 3. heapq: 힙 자료구조를 제공 - 우선순위 큐 기능 구현
# 4. bisect: 이진 탐색(Binary Search) 기능 제공
# 5. collections: 덱(deque), 카운터(counter)
# 6. math: 필수적인 수학 기능: 팩토리얼, 제곱근, GCD, 삼각함수 관련, 파이

# sum()
# result = sum([1, 2, 3, 4, 5])
# print(result)

# min(), max()
# min = min(7, 4, 5, 1, 2)
# max = max(7, 4, 5, 1, 2)
# print(min, max)

# # eval(): 수식으로 표현된 값을 반환
# eval = eval("(3+5)*7")
# print(eval)

# sorted()
# result = sorted([9, 1, 3, 4, 2])
# reverse_result = sorted([9, 1, 3, 4, 2], reverse=True)
# print(result)
# print(reverse_result)

# # sorted() with key
# array = [('홍길동', 35), ('이순신', 75), ('아무개', 50)]
# result = sorted(array, key=lambda x: x[1], reverse=True)
# print(result)


# # 순열
# data = ['A', 'B', 'C']

# result = list(permutations(data, 3))
# print(result)

# # 조합
# data = ['A', 'B', 'C']

# result = list(combinations(data, 3))
# print(result)
