# 다이나믹 프로그래밍 : 메모리를 적절히 사용하여 수행 시간 효율성을 비약적으로 향상시키는 방법
# 이미 계산된 결과 (작은 문제)는 별도의 메모리 영역에 저장하여 다시 계산하지 않도록 한다.
# 다이나믹 프로그래밍의 구현은 일반적으로 두 가지 방식 (탑다운-하향식, 보텀업-상향식)으로 구성된다.

# 다이나믹 프로그래밍은 동적 계획법이라고도 부른다.
# 일반적인 프로그래밍 분야에서의 동적 (Dynamic)이란 어떤 의미를 가질까?
# 자료구조에서 동적 할당은 '프로그램이 실행되는 도중에 실행에 필요한 메모리를 할당하는 기법'을 의미한다.
# 반면에 다이나믹 프로그래밍에서 '다이나믹'은 별다른 의미 없이 사용된 단어이다.

# 다이나믹 프로그래밍은 문제가 다음의 조건을 만족할 때 사용할 수 있다.
# 1. 최적 부분 구조 (optimal substructure) : 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있다.
# 2. 중복되는 부분 문제 (overlapping subproblem) : 동일한 작은 문제를 반복적으로 해결해야 한다.

# 피보나치 수열 : 다이나믹 프로그래밍으로 효과적으로 계산할 수 있다.
# 1,1,2,3,5,8,13,21,34,55,89
# 점화식 : 인접한 항들 사이의 관계식을 의미한다.
# 피보나치 수열을 점화식으로 표현하면 다음과 같다. a(n) = a(n-1) + a(n-2), a1 = 1, a2 = 1
# 프로그래밍에서는 이러한 수열을 배열이나 리스트를 이용해 표현한다

# 피보나치 수열 : 단순 재귀 소스코드

def Fibo(i):
    if i <= 2:
        return 1
    else:
        return Fibo(i-2) + Fibo(i-1)


fibo = int(input())
result = Fibo(fibo)

print(result)


# 피보나치 수열의 시간 복잡도 분석 : 단순 재귀 함수로 피보나치 수열을 해결하면 지수 시간 복잡도를 가지게 된다. - O(2^N)
# 중복되는 부분이 문제가 된다. : f(6)을 구하기 위해서 f(2)를 5번 구해야 한다.
# 빅오 표기법을 기준으로 f(30)을 계산하기 위해 약 10억가량의 연산을 수행해야 한다.

# 피보나치 수열의 효율적인 해법 : 다이나믹 프로그래밍
# 다이나믹 프로그래밍의 사용 조건을 만족하는지 확인한다. : 최적 부분 구조, 중복되는 부분 문제
# 피보나치 수열은 다이나믹 프로그래밍의 사용 조건을 만족한다.

# 메모이제이션 (Memoization) : 다이나믹 프로그래밍을 구현하는 방법 중 하나 (탑다운, 하향식)
# 한 번 계산한 결과를 메모리 공간에 메모하는 기법
# 같은 문제를 다시 호출하면 메모했던 결과를 그대로 가져온다.
# 값을 기록해 놓는다는 점에서 캐싱 (Caching)이라고도 한다.

# 탑다운 VS 보텀업
# 탑다운 (메모이제이션) 방식은 하향식이라고도 하며 보텀업 방식은 상향식이라고도 한다.
# 다이나믹 프로그래밍의 전형적인 형태는 보텀업 방식이다. : 결과 저장용 리스트는 DP 테이블이라고 부른다.
# 엄밀히 말하면 메모이제이션은 이전에 계산된 결과를 일시적으로 기록해 놓는 넓은 개념을 의미한다. 따라서 메모이제이션은 다이나믹 프로그래밍에 국한된 개념은 아니다.
# 한번 계산된 결과를 담아 놓기만 하고 다이나믹 프로그래밍을 위해 활용하지 않을 수도 있다.

# 피보나치 수열: 탑다운 다이나믹 프로그래밍 소스코드

d = [0]*100


def Fibo(x):
    # 종료 조건 (1 혹은 2일 때 1을 반환)
    if x == 1 or x == 2:
        return 1

    # 이미 계산한 적 있는 문제라면 그대로 반환
    if d[x] != 0:
        return d[x]

    # 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환
    d[x] = Fibo(x-1) + Fibo(x-2)
    return d[x]


print(Fibo(99))


# 피보나치 수열 : 보텀업 다이나믹 프로그래밍 소스코드

d = [0] * 100

# 첫번째 피보나치 수와 두번째 피보나치 수는 1
d[1] = 1
d[2] = 1
n = 99

# 피보나치 함수 반복문으로 구현
for i in range(3, n+1):
    d[i] = d[i-1] + d[i-2]

print(d[n])


# 피보나치 수열 : 메모이제이션 동작 분석
# 이미 계산된 결과를 메모리에 저장해 둔다면...
# f(6)을 구하기 위해서 f(5), f(4), f(3)만 계산해두면 된다.

# 메모이제이션을 이용하는 경우 피보나치 수열 함수의 시간 복잡도는 O(N)이다.

d = [0]*100


def Fibo(x):
    print("f (", x, ")", end=' ')
    if x == 1 or x == 2:
        return 1

    if d[x] == 0:
        d[x] = Fibo(x-1) + Fibo(x-2)
        return d[x]
    else:
        return d[x]


Fibo(6)


# 다이나믹 프로그래밍 vs 분할 정복
# 다이나믹 프로그래밍과 분할 정복은 모두 최적 부분 구조를 가질 때 사용할 수 있다. : 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있는 상황
# 다이나믹 프로그래밍과 분할 정복의 차이점은 '부분 문제의 중복'이다.
# 다이나믹 프로그래밍 문제에서는 각 부분 문제들이 서로 영향을 미치며 부분 문제가 중복된다.
# 분할 정복 문제에서는 동일한 부분 문제가 반복적으로 계싼되지 않는다.

# 다이나믹 프로그래밍 문제에 접근하는 방법
# 주어진 문제가 다이나믹 프로그래밍 유형임을 파악하는 것이 중요하다.
# 가장 먼저 그리디, 구현, 완전 탐색 등의 아이디어롤 문제를 해결할 수 있는지 검토할 수 있다.
# 다른 알고리즘으로 풀이 방법이 떠오르지 않으면 다이나믹 프로그래밍을 고려해 본다.
# 일단 재귀 함수로 비효율적인 완전 탐색 프로그램을 작성한 뒤에 (탑다운) 작은 문제에서 구한 답이 큰 문제에서 그대로 사용될 수 있으면, 코드를 개선하는 방법을 사용할 수 있다.
# 일반적인 코딩 테스트 수준에서는 기본 유형의 다이나믹 프로그래밍 문제가 출제되는 경우가 많다.

# < 문제 > 개미 전사
# 개미 전사는 부족한 식량을 충당하고자 메뚜기 마을의 식량창고를 몰래 공격하려고 한다. 메뚜기 마을에는 여러 개의 식량 창고가 있는데 식량창고는 일직선으로 이어져 있다.
# 각 식량창고에는 정해진 수의 식량을 저장하고 있으며 개미 전사는 식량 창고를 선택적으로 약탈하여 식량을 빼앗을 예정이다. 이때 메뚜기 정찰병들은 일직선상에 존재하는 식량창고 중에서 서로
# 인접한 식량창고가 공격받으면 바로 알아챌 수 있다.
# 따라서 개미 전사가 정찰병에게 들키지 않고 식량창고를 약탈하기 위해서는 최소한 한 칸 이상 떨어진 식량창고를 약탈해야 한다.

N = int(input())
foods = list(map(int, input().split()))
sum = 0

# 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
d = [0]*100

# 다이나믹 프로그래밍 진행 (보텀업)
d[0] = foods[0]
d[1] = max(foods[0], foods[1])

for i in range(2, N):
    d[i] = max(d[i-1], d[i-2]+foods[i])

print(d[N-1])


# < 문제 > 1로 만들기
# 정수 X가 주어졌을 때, 정수 X에 사용할 수 있는 연산은 다음과 같이 4가지입니다.
# 1. X가 5로 나누어 떨어지면, 5로 나눕니다.
# 2. X가 3으로 나누어 떨어지면, 3으로 나눕니다.
# 3. X가 2로 나누어 떨어지면, 2로 나눕니다.
# 4. X에서 1을 뺍니다.
# 정수 X가 주어졌을 때, 연산 4개를 적절히 사용해서 값을 1로 만들고자 합니다. 연산을 사용하는 횟수의 최솟값을 출력하세요. 예를 들어 정수가 26이면 다음과 같이 계싼해서 3번의 연산이 최솟값입니다.
# 26 -> 25 -> 5 -> 1

# 이 문제는 나누기를 많이 한다고 해서 연산이 최소가 되는 것이 아니다. 네가지의 연산을 적절하게 조화해서 더 작게 만드는 것이 포인트.
# 피보나치의 트리 구조를 다시 한번 생각해본다.
x = int(input())
count = 0

while True:
    if x % 30 != 0:
        x -= 1
        count += 1

print(count)

# a_i는 i를 1로 만들기 위한 최소 연산 횟수
# 점화식은 다음과 같다.
# a_i = min(a_i-1, a_i/2, a_i/3, a_i/5) + 1
# 단, 1을 빼는 연산을 제외하고는 해당 수로 나누어질 때에 한해 점화식을 적용할 수 있다.

x = int(input())
d = [0] * 30001

# 다이나믹 프로그래밍 진행 (보텀업)
for i in range(2, x+1):
    # 현재의 수에서 1을 빼는 경우
    d[i] = d[i-1] + 1

    # 현재의 수가 2로 나누어 떨어지는 경우
    if i % 2 == 0:
        d[i] = min(d[i], d[i//2] + 1)

    # 현재의 수가 3으로 나누어 떨어지는 경우
    if i % 3 == 0:
        d[i] = min(d[i], d[i//3] + 1)

    # 현재의 수가 5으로 나누어 떨어지는 경우
    if i % 5 == 0:
        d[i] = min(d[i], d[i//5] + 1)

print(d[x])


# < 문제 > 효율적인 화폐 구성
# N가지 종류의 화폐가 있습니다. 이 화폐들의 개수를 최소한으로 이용해서 그 가치의 합이 M원이 되도록 하려고 합니다. 이때 각 종류의 화폐는 몇 개라도 사용할 수 있습니다.
# 예를 들어 2원, 3원 단위의 화폐가 있을 때에는 15원을 만들기 위해 3원을 5개 사용하는 것이 가장 최소한의 화폐 개수입니다.
# M원을 만들기 위한 최소한의 화폐 개수를 출력하는 프로그램을 작성하세요.

# a_i : 금액 i를 만들 수 있는 최소한의 화폐 개수
# k : 각 화폐의 단위
# 점화식 : 각 화폐 단위인 k를 하나씩 확인하며
# a_i-k를 만드는 방법이 존재하는 경우, a_i = min(a_i, a_i-k + 1)
# a_i-k를 만드는 방법이 존재하지 않는 경우, a_i = INF

n, m = map(int, input().split())
array = []

for i in range(n):
    array.append(int(input()))

dp = [10001] * (m+1)

for i in range(n):
    for j in range(array[i], m+1):
        if dp[j-array[i]] != 10001:
            dp[j] = min(dp[j], dp[j - array[i]]+1)

if dp[m] == 10001:
    print(-1)
else:
    print(dp[m])


# <문제> 금광
# n X m 크기의 금광이 있습니다. 금광은 1 X 1 크기의 칸으로 나누어져 있으며, 각 칸은 특정한 크기의 금이 들어있습니다.
# 채굴자는 첫번째 열부터 출발하여 금을 캐기 시작합니다. 맨 처음에는 첫번째 열의 어느 행에서든 출발할 수 있습니다.
# 이후에 m-1번에 걸쳐서 매번 오른쪽 위, 오른쪽, 오른쪽 아래 3가지 중 하나의 위치로 이동해야 합니다.
# 결과적으로 채굴자가 얻을 수 있는 금의 최대 크기를 출력하는 프로그램을 작성하세요.

for tc in range(int(input())):
    # 금과 정보 입력
    n, m = map(int, input().split())
    array = list(map(int, input().split()))
    
    dp = []
    index = 0
    for i in range(n):
        dp.append(array[index:index+m])
        index += m
    
    for j in range(1, m):
        for i in range(n):
            if i == 0:
                left_up = 0
            else : 
                left_up = dp[i-1][j-1]
                
            if i == n-1:
                left_down = 0
            else :
                 left_down = dp[i+1][j-1]
                 
            left = dp[i][j-1]
            dp[i][j] = dp[i][j] + max(left_up, left_down,left)
    result = 0
    for i in range(n):
        result = max (result, dp[i][m-1])
    
    print(result)

   
# <문제> 병사 배치하기
# N명의 병사가 무작위로 나열되어 있습니다. 각 병사는 특정한 값의 전투력을 보유하고 있습니다. 병사를 배치할 때에는 전투력이 높은 병사가 앞쪽에 오도록 내림차순으로 배치를 하고자 합니다. 
# 다시 말해 앞쪽에 있는 병사의 전투력이 항상 뒤쪽에 있는 병사보다 높아야 합니다.
# 또한 배치 과정에서는 특정한 위치에 있는 병사를 열외시키는 방법을 이용합니다. 그러면서도 남아 있는 병사의 수가 최대가 되도록 하고 싶습니다.
# 병사에 대한 정보가 주어졌을 때, 남아 있는 병사의 수가 최대가 되도록 하기 위해서 열외시켜야 하는 병사의 수를 출력하는 프로그램을 작성하세요.

n = int(input())
array = list(map(int, input().split()))
array.reverse()

dp = [1] * n

for i in range(1, n):
    for j in range(0,i):
        if array[j] < array[i]:
            dp[i] = max(dp[i], dp[j] + 1)
            
print(n - max(dp))
