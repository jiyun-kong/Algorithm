# -- coding: utf-8 --

# �׸��� �˰����� (Ž���) : ���� ��Ȳ���� ���� ���� ���� �͸� ������ ���
# ������ Ǯ�� ���� �ּ����� ���̵� ���ø� �� �ִ� �ɷ� �䱸
# ���缺 �м��� ���� �߿��ϴ� : �ܼ��� ���� ���� ���̴� ���� �ݺ������� �����ص� ������ �ظ� ���� �� �ִ��� �����Ѵ�.
# �Ϲ����� ��Ȳ���� �׸��� �˰������� ������ �ظ� ������ �� ���� ���� ����.
# ������ ���׿����� ��κ��� �׸��� ������ Ž������� ���� �ذ� ������ �ذ� �Ǵ� ��Ȳ����, �̸� �߷��� �� �־�� Ǯ������ �����ȴ�.

# <����> �Ž��� ��
# ����� �������� ����� �����ִ� �����̴�. ī���Ϳ��� �Ž��������� ����� 500��, 100��, 50��, 10��¥�� ������ ������ �����Ѵٰ� �����Ѵ�.
# �մԿ��� �Ž��� �־�� �� ����  N���� ��, �Ž��� �־�� �� ������ �ּ� ������ ���Ͻÿ�. ��, �Ž��� ��� �� �� N�� �׻� 10�� ����̴�.

# <���缺 �м�>
# ���� ū ȭ�� �������� ���� �Ž��� �ִ� ���� ������ �ظ� �����ϴ� ������?
# ������ �ִ� ���� �߿��� ū ������ �׻� ���� ������ ����̹Ƿ� ���� �����l �������� ������ �ٸ� �ذ� ���� �� ���� ����
# ���࿡ 800���� �Ž��� �־�� �ϴµ� ȭ�� ������ 500��, 400��, 100���̶��? 400�� ������ 2�� �Ž��� �ִ� ���� ������.

# ���� ���̸� ���� �ּ����� ���̵� ���ø��� �̰��� �������� ������ �� �־�� �Ѵ�.


n = 1260
coin = [500, 100, 50, 10]
numbers = 0

for i in coin:
    numbers += n // i
    n = n % i

print(numbers)


# �ð� ���⵵ �м�
# ȭ�� ������ K��� �� ��, �ҽ��ڵ��� �ð� ���⵵�� O(K)
# �Ž����־��� �ϴ� �ݾװ��� �����ϸ�, ������ �� �������� ������ ����


# <����> 1�� �� ������
# ��� �� N�� 1�� �� ������ ������ �� ���� �� �ϳ��� �ݺ������� �����Ͽ� �����Ϸ��� �Ѵ�. ��, �ι�° ������ N�� K�� ������ ������ ���� ������ �� �ִ�.
# 1. N���� 1�� ����.
# 2. N�� K�� ������.

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


# �־��� N�� ���Ͽ� �ִ��� ���� �����⸦ �����ϸ� �ȴ�.
# N�� ���� ���� �� 2 �̻��� ���� ������ �۾��� 1�� ���� �۾����� ���� �ξ� ���� ���� �� �ִ�.

# <���缺 �м�>
# �����ϸ� �ִ��� ���� ������ �۾��� ������ �ظ� �׻� ������ �� ������?
# N�� �ƹ��� ū ������, K�� ��� �����ٸ� ���ϱ޼������� ������ ���� �� �ִ�.
# �ٽ� ����, K�� 2 �̻��̱⸸ �ϸ�, K�� ������ ���� 1�� ���� �ͺ��� �׻� ������ N�� ���� �� �ִ�.
# ���� N�� �׻� 1�� �����ϰ� �ȴ�. (������ �� ����)


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


# �ð� ���⵵ : O(logN)


# <����> ���ϱ� Ȥ�� ���ϱ�
# �� �ڸ��� ���� (0 ~ 9)�θ� �̷���� ���ڿ� S�� �־����� ��, ���ʺ��� ���������� �ϳ��� ��� ���ڸ� Ȯ���ϸ� ���� ���̿� 'X' Ȥ�� '+' �����ڸ� �־� ��������� ������� �� �ִ�
# ���� ū ���� ���ϴ� ���α׷��� �ۼ��Ͻÿ�.
# ��, +���� X�� ���� ����ϴ� �Ϲ����� ��İ��� �޸�, ��� ������ ���ʿ������� ������� �̷�����ٰ� �����Ѵ�.


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


# ��κ��� ��� �������ٴ� ������ ���� �� ũ�� �����.
# �ٸ�, �� �� �߿��� �ϳ��� 0�̰ų� 1�� ���, ���ϱ⺸�ٴ� ���ϱ⸦ �����ϴ� ���� ȿ�����̴�.
# ����, �� ���� ���Ͽ� ������ ������ ��, �� �� �߿��� �ϳ��� 1 ������ ��쿡�� ���ϸ�, �� ���� ��� 2 �̻��� ��쿡�� ���ϸ� �����̴�.

data = input()

result = int(data[0])

for i in range(1, len(data)):
    num = int(data[i])
    if num <= 1 or result <= 1:
        result += num
    else:
        result *= num

print(result)