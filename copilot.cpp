#include <bits/stdc++.h>
// #include <queue>
// #include <vector>
// #include <cmath> // Thêm thư viện cmath để sử dụng hàm sqrt
using namespace std;
const double INF = numeric_limits<double>::infinity();
const int M = 20;
const int N = 20;
const int adjSize = M * N;

vector<vector<double>> grid(M, vector<double>(N, 0));              // Sử dụng kiểu double cho grid
vector<vector<double>> adj(adjSize, vector<double>(adjSize, 0.0)); // Sử dụng kiểu double cho adj

void setGrid();
void displayGrid();
void setAdj();
void displayAdj();
void setWall(vector<vector<double>> &adj, int wallNode);
void resetWalls();
void BFS(vector<vector<double>> &matrix, int start, int end);
void Dijkstra(vector<vector<double>> &matrix, int start, int end);
double euclideanHeuristic(int u, int end);
void AStar(vector<vector<double>> &graph, int start, int end);
int main()
{
    setGrid();
    // displayGrid();
    setAdj();
    // displayAdj();
    setWall(adj, 21);
    setWall(adj, 41);
    setWall(adj, 61);
    setWall(adj, 81);
    setWall(adj, 3);
    setWall(adj, 23);
    setWall(adj, 43);
    setWall(adj, 63);
    setWall(adj, 83);
    setWall(adj, 148);
    setWall(adj, 167);
    setWall(adj, 186);
    setWall(adj, 205);
    resetWalls();
    // cout << '\n';
    // displayAdj();
    cout << "BFS: \n";
    BFS(adj, 0, 168); // Chỉ số bắt đầu từ 0 đến 24
    cout << '\n';
    cout << "Dijkstra: \n";
    Dijkstra(adj, 0, 168);
    cout << "A star: \n";
    AStar(adj, 0, 168);
    return 0;
}

void setGrid()
{
    double value = 0; // Bắt đầu từ 0
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i][j] = value;
            value++;
        }
    }
}

void displayGrid()
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << grid[i][j] << ' ';
        }
        cout << '\n';
    }
}

void setAdj()
{
    int dirI[8] = {-1, -1, -1, 0, 0, 1, 1, 1}; // Thêm hướng cho đường chéo
    int dirJ[8] = {-1, 0, 1, -1, 1, -1, 0, 1}; // Thêm hướng cho đường chéo
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < 8; k++) // Lặp qua 8 hướng
            {
                int ni = i + dirI[k];
                int nj = j + dirJ[k];
                if (ni >= 0 && ni < M && nj >= 0 && nj < N)
                {
                    double distance = (dirI[k] != 0 && dirJ[k] != 0) ? sqrt(2) : 1; // Khoảng cách 1.41 cho đường chéo
                    adj[grid[i][j]][grid[ni][nj]] = distance;
                    adj[grid[ni][nj]][grid[i][j]] = distance;
                }
            }
        }
    }
}

void displayAdj()
{
    for (int i = 0; i < adjSize; i++)
    {
        for (int j = 0; j < adjSize; j++)
        {
            cout << adj[i][j] << ' ';
        }
        cout << '\n';
    }
}
void setWall(vector<vector<double>> &adj, int wallNode)
{
    // Đặt tất cả các kết nối từ và đến nút tường thành 0
    for (int i = 0; i < adjSize; i++)
    {
        if (adj[wallNode][i] != 0)
        {
            adj[wallNode][i] = 0.0;
            adj[i][wallNode] = 0.0;
        }
    }
}

// Hàm BFS cần được chỉnh sửa để xử lý kiểu double và khoảng cách đường chéo
// ...

void BFS(vector<vector<double>> &matrix, int start, int end)
{
    vector<bool> visited(matrix.size(), false);
    vector<int> queue;
    vector<int> parent(matrix.size(), -1);
    queue.push_back(start);
    visited[start] = true;

    int vis, i, curr;
    while (!queue.empty())
    {
        vis = queue[0];

        cout << vis << " ";

        queue.erase(queue.begin());
        for (i = 0; i < matrix[vis].size(); i++)
        {
            if (matrix[vis][i] != 0 && (!visited[i]))
            {
                queue.push_back(i);
                visited[i] = true;
                parent[i] = vis;
            }
        }
    }

    cout << "\n Path from " << start << " to " << end << " : ";
    curr = end;
    while (curr != -1)
    {
        cout << curr;
        if (curr != start)
        {
            cout << " <- ";
        }
        curr = parent[curr];
    }
}
void Dijkstra(vector<vector<double>> &matrix, int start, int end)
{
    vector<double> dist(matrix.size(), INF);
    vector<int> parent(matrix.size(), -1);
    priority_queue<pair<double, int>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty())
    {
        int u = pq.top().second;
        double d = -pq.top().first;
        pq.pop();

        if (d > dist[u])
        {
            continue; // Bỏ qua nếu đã có đường đi ngắn hơn
        }

        for (int v = 0; v < matrix[u].size(); ++v)
        {
            double w = matrix[u][v];
            if (w != 0 && dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({-dist[v], v});
            }
        }
    }

    // In ra khoảng cách và đường đi từ đỉnh start đến end
    cout << "Distance from " << start << " to " << end << ": " << dist[end] << endl;
    cout << "Path: ";
    int node = end;
    while (node != -1)
    {
        cout << node;
        if (node != start)
        {
            cout << " <- ";
        }
        node = parent[node];
    }
    cout << endl;
}
double euclideanHeuristic(int u, int end)
{
    int ui = u / N;
    int uj = u % N;
    int endi = end / N;
    int endj = end % N;
    return sqrt((ui - endi) * (ui - endi) + (uj - endj) * (uj - endj));
}

void AStar(vector<vector<double>> &matrix, int start, int end)
{
    vector<double> dist(matrix.size(), INF);
    vector<int> parent(matrix.size(), -1);
    priority_queue<pair<double, int>> pq;
    double vision = euclideanHeuristic(start, end);
    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty())
    {
        int u = pq.top().second;
        double d = -pq.top().first;
        pq.pop();

        if (d > dist[u])
        {
            continue; // Bỏ qua nếu đã có đường đi ngắn hơn
        }

        for (int v = 0; v < matrix[u].size(); ++v)
        {
            double w = matrix[u][v];
            if (w != 0 && dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({-(vision - euclideanHeuristic(v, end)), v}); // Thêm giá trị euclideanHeuristic
            }
        }
    }

    // In ra khoảng cách và đường đi từ đỉnh start đến end
    cout << "Distance from " << start << " to " << end << ": " << dist[end] << endl;
    cout << "Path: ";
    int node = end;
    while (node != -1)
    {
        cout << node;
        if (node != start)
        {
            cout << " <- ";
        }
        node = parent[node];
    }
    cout << endl;
}
void resetWalls()
{
    for (int i = 0; i < adjSize; i++)
    {
        if (i + N < adjSize && i + 1 < adjSize && adj[i][i + N] == 0 && adj[i][i + 1] == 0)
        {
            adj[i][i + N + 1] = 0;
            adj[i + N + 1][i] = 0;
        }
        else if (i + N < adjSize && i - 1 >= 0 && adj[i][i + N] == 0 && adj[i][i - 1] == 0)
        {
            adj[i][i + N - 1] = 0;
            adj[i + N - 1][i] = 0;
        }
        else if (i - N >= 0 && i - 1 >= 0 && adj[i][i - N] == 0 && adj[i][i - 1] == 0)
        {
            adj[i][i - N - 1] = 0;
            adj[i - N - 1][i] = 0;
        }
        else if (i - N >= 0 && i + 1 < adjSize && adj[i][i - N] == 0 && adj[i][i + 1] == 0)
        {
            adj[i][i - N + 1] = 0;
            adj[i - N + 1][i] = 0;
        }
    }
}
