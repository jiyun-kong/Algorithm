# BFS review
N, M = map(int, input().split())
iceBox = []
count = 0

for n in range(N):
    iceBox.append(list(input()))


def iceCounting(row_i, column_j):
    if row_i < 0 or row_i >= N or column_j < 0 or column_j >= M:
        return False

    if iceBox[row_i][column_j] == '1':
        return False
    else:
        iceBox[row_i][column_j] = '1'
        iceCounting(row_i - 1, column_j)
        iceCounting(row_i + 1, column_j)
        iceCounting(row_i, column_j - 1)
        iceCounting(row_i, column_j + 1)
        return True


for i in range(N):
    for j in range(M):
        if iceCounting(i, j) == True:
            count += 1


print(count)


# DFS review
n, m = map(int, input().split())
maze = []
count = 0

for _ in range(n):
    maze.append(list(map(int, input())))


def maze_exit(i, j):
    global count

    if i >= n or i < 0 or j >= n or j < 0:
        return False

    if maze[i][j] == 0:
        return False
    else:
        count += 1
        maze[i][j] = 0
        maze_exit(i+1, j)
        maze_exit(i, j+1)
        return True


maze_exit(0, 0)

print(count)
