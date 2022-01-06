# 탑 다운 (메모이제이션) 방식으로 피보나치 수열 구하기 : 연산이 진행된 적이 있는 요소와 그렇지 않은 요소의 구분!!!
dp = [0] * 1001


def fibo(n):
    if n == 1 or n == 2:
        return 1

    if dp[n] == 0:      # 아직 연산된 적이 없음. 연산 필요하다.
        dp[n] = fibo(n-1) + fibo(n-2)
        return dp[n]
    elif dp[n] != 0:    # 연산된 적이 있는 상태임.
        return dp[n]


num = int(input())
print(fibo(num))


# 바텀 업 방식으로 피보나치 수열 구하기 : 가장 작은 원소에서부터 하나씩 구해서 목표 원소까지 도달하기!!
dp = [0] * 1001


def fibo(n):
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
        print("dp[", i, "] = ", dp[i])

    return dp[n-1]


num = int(input())


# <금광> 문제 다시 풀기
mine = []

for t in range(int(input())):
    n, m = map(int, input().split())
    array = list(map(int, input().split()))
    dp = []

    # 1차원 배열을 2차원 배열로 만들기
    idx = 0
    for i in range(n):
        dp.append(array[idx:idx+m])
        idx = idx + m

    for j in range(1, m):
        for i in range(n):

            if i == 0:
                left_up = 0
            else:
                left_up = dp[i-1][j-1]

            if i == (n-1):
                left_down = 0
            else:
                left_down = dp[i+1][j-1]

            left = dp[i][j-1]
            dp[i][j] = dp[i][j] + max(left_up, left_down, left)

    # m-1 열 중에서 가장 큰 값 구하기
    result = 0
    for i in range(n):
        result = max(result, dp[i][m-1])

    print(result)

print(fibo(num))


# <개미 전사> 문제 다시 풀기
n = int(input())                        # 식량 창고 개수
k = list(map(int, input().split()))      # 식량 창고에 저장된 식량의 개수 리스트

dp = [0] * 100

dp[0] = k[0]
dp[1] = max(k[0], k[1])

for i in range(2, n):
    dp[i] = max(dp[i-2]+k[i], dp[i-1])

print(dp[n-1])


# <1로 만들기> 문제 다시 풀기
'''
x = int(input())
dp = [0] * 30001


def makeOne(x):
    if dp[x] != 0:      # 이미 연산이 진행된 요소
        return dp[x]
    else:               # 연산이 필요한 요소
        if x % 2 == 0:
            dp[x] = min(dp(x-1), makeOne(x//2)) + 1
        elif x % 3 == 0:
            dp[x] = min(makeOne(x-1), makeOne(x//3)) + 1
        elif x % 5 == 0:
            dp[x] = min(makeOne(x-1), makeOne(x//5)) + 1
        else:
            dp[x] = makeOne(x-1) + 1

        return dp[x]


print(makeOne(x))
'''

# 탑다운 방식으로 접근하려 했지만 무한 루프를 돌면서 실패했다.
# 피보나치 수열에서 f(1), f(2), f(3), ... 차례대로 다 구하고 메모이제이션 했던 것처럼 이 문제도 f(1), f(2) ... 하나씩 구하는 바텀업 방식을 사용해본다.

x = int(input())
dp = [0] * 30001

dp[1] = 0
dp[2] = 1

for i in range(3, x+1):
    # if ~ elif로 하면 안된다. 동시에 만족하는 경우 (i == 30)도 있을 수 있어서! 갱신의 느낌으로다가... 가야 한다!
    # 자신의 값에서 1 빼기 연산은 모든 요소가 공통적으로 할 수 있는 연산이기 때문에 그 값으로 초기화 해 주고 특수한 경우 (2, 3, 5로 나눠 떨어지는 경우)와 비교한다.

    dp[i] = dp[i-1] + 1

    if i % 2 == 0:
        dp[i] = min(dp[i//2] + 1, dp[i])
    if i % 3 == 0:
        dp[i] = min(dp[i//3] + 1, dp[i])
    if i % 5 == 0:
        dp[i] = min(dp[i//5] + 1, dp[i])

print(dp[x])


# <효율적인 화폐 구성> 문제 다시 풀기
n, m = map(int, input().split())    # n : 화폐 가치 종류 개수, m : 목표 금액
value = []
dp = [10001] * (m+1)
dp[0] = 0

for nn in range(n):
    value.append(int(input()))

for i in range(n):      # 화폐 종류만큼 반복
    for j in range(value[i], m+1):
        if dp[j - value[i]] != 10001:
            dp[j] = min(dp[j], dp[j - value[i]] + 1)

if dp[m] == 10001:
    print(-1)
else:
    print(dp[m])
