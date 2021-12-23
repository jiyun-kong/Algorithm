# -- coding: utf-8 --

# 그리디 알고리즘 (탐욕법) : 현재 상황에서 지금 당장 좋은 것만 고르는 방법
# 문제를 풀기 위한 최소한의 아이디어를 떠올릴 수 있는 능력 요구
# 정당성 분석이 가장 중요하다 : 단순히 가장 좋아 보이는 것을 반복적으로 선택해도 최적의 해를 구할 수 있는지 검토한다.
# 일반적인 상황에서 그리디 알고리즘은 최적의 해를 보장할 수 없을 때가 많다.
# 하지만 코테에서의 대부분의 그리디 문제는 탐욕법으로 얻은 해가 최적의 해가 되는 상황에서, 이를 추론할 수 있어야 풀리도록 출제된다.

# <문제> 거스름 돈
# 당신은 음식점의 계산을 도와주는 점원이다. 카운터에는 거스름돈으로 사용할 500원, 100원, 50원, 10원짜리 동전이 무한히 존재한다고 가정한다.
# 손님에게 거슬러 주어야 할 돈이  N원일 때, 거슬러 주어야 할 동전의 최소 개수를 구하시오. 단, 거슬러 줘야 할 돈 N은 항상 10의 배수이다.

# <정당성 분석>
# 가장 큰 화폐 단위부터 돈을 거슬러 주는 것이 최적의 해를 보장하는 이유는?
# 가지고 있는 동전 중에서 큰 단위가 항상 작은 단위의 배수이므로 작은 단위릐 동전들을 종합해 다른 해가 나올 수 없기 때문
# 만약에 800원을 거슬러 주어야 하는데 화폐 단위가 500원, 400원, 100원이라면? 400원 동전을 2개 거슬러 주는 것이 최적임.

# 문제 출이를 위한 최소한의 아이디어를 떠올리고 이것이 정당한지 검토할 수 있어야 한다.


n = 1260
coin = [500, 100, 50, 10]
numbers = 0

for i in coin:
    numbers += n // i
    n = n % i

print(numbers)


# 시간 복잡도 분석
# 화폐 종류가 K라고 할 때, 소스코드의 시간 복잡도는 O(K)
# 거슬러주어햐 하는 금액과는 무관하며, 동전의 총 종류에만 영향을 받음


# <문제> 1이 될 때까지
# 어떠한 수 N이 1이 될 때까지 다음의 두 과정 중 하나를 반복적으로 선택하여 수행하려고 한다. 단, 두번째 연산은 N이 K로 나누어 떨어질 때만 선택할 수 있다.
# 1. N에서 1을 뺀다.
# 2. N을 K로 나눈다.

N, K = input().split()
N = int(N)
K = int(K)

nums = 0

while N != 1:
    if N % K == 0:
        nums += 1
        N = N // K
    else:
        nums += 1
        N = N - 1

print(nums)


# 주어진 N에 대하여 최대한 많이 나누기를 수행하면 된다.
# N의 값을 줄일 때 2 이상의 수로 나누는 작업이 1을 빼는 작업보다 수를 훨씬 많이 줄일 수 있다.

# <정당성 분석>
# 가능하면 최대한 많이 나누는 작업이 최적의 해를 항상 보장할 수 있을까?
# N이 아무리 큰 수여도, K로 계속 나눈다면 기하급수적으로 빠르게 줄일 수 있다.
# 다시 말해, K가 2 이상이기만 하면, K로 나누는 것이 1을 빼는 것보다 항상 빠르게 N을 줄일 수 있다.
# 또한 N은 항상 1에 도달하게 된다. (최적의 해 성립)


N, K = map(int, input().split())
result = 0

while True:
    target = (N // K) * K
    result += (N - target)
    N = target

    if N < K:
        break
    result += 1
    N //= K

result += (n-1)
print(result)


# 시간 복잡도 : O(logN)


# <문제> 곱하기 혹은 더하기
# 각 자리가 숫자 (0 ~ 9)로만 이루어진 문자열 S가 주어졌을 때, 왼쪽부터 오른쪽으로 하나씩 모든 숫자를 확인하며 숫자 사이에 'X' 혹은 '+' 연산자를 넣어 결과적으로 만들어질 수 있는
# 가장 큰 수를 구하는 프로그램을 작성하시오.
# 단, +보다 X를 먼저 계산하는 일반적인 방식과는 달리, 모든 연산은 왼쪽에서부터 순서대로 이루어진다고 가정한다.


N = list(input())
result = 0
N = list(map(int, N))


for i in range(len(N) - 1):
    if i == 0:
        if N[i] == 0 or N[i] == 1:
            result = result + N[i+1]
        else:
            result = 1
            result = result * N[i+1]
    else:
        if N[i] == 0 or N[i] == 1:
            result = result + N[i+1]
        else:
            result = result * N[i+1]

print(result)


# 대부분의 경우 덧셈보다는 곱셈이 값을 더 크게 만든다.
# 다만, 두 수 중에서 하나라도 0이거나 1인 경우, 곱하기보다는 더하기를 수행하는 것이 효율적이다.
# 따라서, 두 수에 대하여 연산을 수행할 때, 두 수 중에서 하나라도 1 이하인 경우에는 더하며, 두 수가 모두 2 이상인 경우에는 곱하면 정답이다.

data = input()

result = int(data[0])

for i in range(1, len(data)):
    num = int(data[i])
    if num <= 1 or result <= 1:
        result += num
    else:
        result *= num

print(result)
