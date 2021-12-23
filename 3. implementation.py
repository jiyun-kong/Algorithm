# -- coding: utf-8 --

# 구현 (implementation) : 머릿속에 있는 알고리즘을 소스코드로 바꾸는 과정
# 흔히 알고리즘 대회에서 구현 유형의 문제란 무엇을 의미할까?
# 풀이를 떠올리는 것은 쉽지만, 소스코드로 옮기기 어려운 문제를 지칭한다.

# 구현 유형의 예시
# 알고리즘은 간단한데, 코드가 지나칠 만큼 길어지는 문제
# 실수 연산을 다루고, 특정 소수점 자리까지 출력해야 하는 문제
# 문자열을 특정한 기준에 따라서 끊어 처리해야 하는 문제
# 적절한 라이브러리를 찾아서 사용해야 하는 문제

# 행렬(Matrix)

for i in range(5):
    for j in range(5):
        print("(", i, ", ", j, ")", end=" ")
    print()


# 시뮬레이션 및 완전 탐색 문제에서는 2차원 공간에서의 방향 벡터가 자주 활용된다.

# 동, 북, 서, 남
dx = [0, -1, 0, 1]   # 상하
dy = [1, 0, -1, 0]   # 좌우

# 현재 위치
x, y = 2, 2

for i in range(4):
    # 다음 위치
    nx = x + dx[i]
    ny = y + dy[i]

    print(nx, ny)


# <문제> 상하좌우
# 여행가 A는 N X N 크기의 정사각형 공간 위에 서 있다. 이 공간은 1 X 1 크기의 정사각형으로 나누어져 있다. 가장 왼쪽 위 좌표는 (1,1)이며, 가장 오른쪽 아래 좌표는 (N,N)에 해당한다.
# 여행가 A는 상, 하, 좌, 우 방향으로 이동할 수 있으며, 시작 좌표는 항상 (1,1)이다. 우리 앞에는 여행가 A가 이동할 계획이 적힌 계획서가 놓여 있다.
# 계획서에는 하나의 줄에 띄어쓰기를 기준으로 하여 L, R, U, D 중 하나의 문자가 반복적으로 적혀 있다. 각 문자의 의미는 다음과 같다.
# L : 왼쪽으로 한 칸 이동
# R : 오른쪽으로 한 칸 이동
# U : 위로 한 칸 이동
# D : 아래로 한 칸 이동
# 이때 여행가 A가 N X N 크기의 정사각형 공간을 벗어나는 움직임은 무시된다. 예를 들어 (1, 1)의 위치에서 L 혹은 U를 만나면 무시된다.


N = int(input())
plan = list(input().split(" "))
loc_lst = [1, 1]

for i in plan:
    if i == 'L':
        if loc_lst[1] != 1:
            loc_lst[1] -= 1
    elif i == 'R':
        if loc_lst[1] != N:
            loc_lst[1] += 1
    elif i == 'U':
        if loc_lst[0] != 1:
            loc_lst[0] -= 1
    elif i == 'D':
        if loc_lst[0] != N:
            loc_lst[0] += 1

print(loc_lst[0], loc_lst[1])


# 이 문제는 요구사항대로 충실히 구현하면 되는 문제이다.
# 일련의 명령에 따라서 개체를 차례대로 이동시킨다는 점에서 시뮬레이션 유형으로도 분류되며 구현이 중요한 대표적인 문제 유형이다.
# 다만, 알고리즘 교재나 문제 풀이 사이트에 따라서 다르게 일컬을 수 있으므로, 코테에서의 시뮬레이션 유형, 구현 유형, 완전 탐색 유형은 서로 유사한 점이 많다는 정도로만 기억하자.


n = int(input())
x, y = 1, 1
plans = input().split()

dx = [0, -1, 0, 1]   # 상하
dy = [1, 0, -1, 0]   # 좌우
move_types = ['L', 'R', 'U', 'D']

for plan in plans:
    for i in range(len(move_types)):
        if plan == move_types[i]:
            nx = x + dx[i]
            ny = y + dy[i]

        if nx < 1 or ny < 1 or nx < n or ny < n:
            continue

        x, y = nx, ny

print(x, y)


# <문제> 시각
# 정수 N이 입력되면 00시 00분 00초부터 N시 59분 59초까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램을 작성하시오.
# 예를 들어 1을 입력했을 때 다음은 3이 하나라도 포함되어 있으므로 세어야 하는 시각이다.
# 00시 00분 03초 / 00시 13분 30초
# 반면에 다음은 3이 하나도 포함되어 있지 않으므로 세면 안되는 시각이다.
# 00시 02분 55초 / 01시 27분 45초

N = int(input())
count = 0

if 0 <= N and N < 3:
    count = (N+1)*1575
elif N >= 3 and N <= 12:
    count = N*1575 + 3600
elif N > 12 and N <= 22:
    count = (N-1)*1575 + 3600*2
elif N == 23:
    count = (N-2)*1575 + 3600*3

print(count)


# 이 문제는 가능한 모든 시각의 경우를 하나씩 모두 세서 풀 수 있는 문제임
# 하루는 86400초이므로, 00사 00분 00초부터 23시 59분 59초까지의 모든 경우는 86400가지이다.
# 파이썬은 1초에 약 2000만번의 연산을 수행한다.
# 따라서 단순히 시각을 1씩 증가시키면서 3이 하나라도 포함되어 있는지를 확인하면 된다.
# 이러한 유형은 완전 탐색 (Brute Forcing) 문제 유형이라고 불린다.
# 가능한 경우의 수를 모두 검사해부는 탐색 방법을 의미한다.


h = int(input())

count = 0
for i in range(h+1):
    for j in range(60):
        for k in range(60):
            if '3' in str(i) + str(j) + str(k):
                count += 1


# <문제> 왕실의 나이트
# 행복 왕국의 왕실 정원은 체스판과 같은 8 x 8 좌표 평면이다. 왕실 정원의 특정한 한 칸에 나이트가 서 있다. 나이트는 매우 충성스러운 신하로서 매일 무술을 연마한다.
# 나이트는 말을 타고 있기 때문에 이동을 할 때에는 L자 형태로만 이동할 수 있으며, 정원 밖으로는 나갈 수 없다.
# 나이트는 특정 위치에서 다음과 같은 2가지 경우로 이동할 수 있다.
# 1. 수평으로 두 칸 이동한 뒤에 수직으로 한 칸 이동하기
# 2. 수직으로 두 칸 이동한 뒤에 수평으로 한 칸 이동하기


loc = list(input())
loc[0] = ord(loc[0]) - ord('a')
loc[1] = int(loc[1])

count = 0

move = [(2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (-1, 2), (1, -2), (-1, -2)]

for mv in move:
    pyeong = loc[0]+mv[0]
    jik = loc[1]+mv[1]

    if pyeong >= 1 and pyeong <= 8:
        if jik >= 1 and jik <= 8:
            count += 1

print(count)


# 요구사항대로 충실히 구현하면 되는 문제이다.
# 나이트의 8가지 경로를 하나씩 확인하며 각 위치로 이동이 가능한지 확인한다.
# 리스트를 이용하여 8가지 방향에 대한 방향 벡터를 정의한다.

input_data = input()
row = int(input_data[1])
column = int(ord(input_data[0])) - int(ord('a')) + 1

steps = [(2, 1), (2, -1), (-2, -1), (-2, 1),
         (1, 2), (-1, 2), (1, -2), (-1, -2)]

result = 0

for step in steps:
    next_row = row + step[0]
    next_column = column + step[1]

    if next_row >= 1 and next_row <= 8 and next_column >= 1 and next_column <= 8:
        result += 1

print(result)

# <문제> 문자열 재정렬
# 알파벳 대문자와 숫자 (0 ~ 9)로만 구성된 문자열이 입력으로 주어진다. 이때 모든 알파벳을 오름차순으로 정렬하여 이어서 출력한 뒤에, 그 뒤에 모든 숫자를 더한 값을 이어서 출력한다.
# 예를 들어 K1KA5CB7이라는 값이 들어오면 ABCKK13을 출력한다.


S = list(input())
alpha = []
number = []
sum = 0

for ch in S:
    if ch >= 'A' and ch <= 'Z':
        alpha.append(ch)
    else:
        number.append(ch)

alpha.sort()

for al in alpha:
    print(al, end='')

for num in number:
    num = int(num)
    sum += num

print(sum)

# 요구사항대로 충실히 구현하면 되는 문제이다.
# 문자열이 입력되었을 때 문자를 하나씩 확인한다,
# 숫자인 경우 따로 합계를 계산한다.
# 알파벳인 경운 별도의 리스트에 저장한다.
# 결과적으로 리스트에 저장된 알파벳을 정렬해 출력하고, 합계를 뒤에 붙여 출력하면 정답이다.


data = input()
result = []
value = 0

for x in data:
    if x.isalpha():
        result.append(x)
    else:
        value += int(x)

result.sort()

if value != 0:
    result.append(str(value))

print(''.join(result))
