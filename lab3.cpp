#include <chrono>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;

const int BOARD_SIZE = 8;
const int TEST_COUNT = 100000;

struct Cell {
    int row;
    int col;
};

class ArrayQueue {
private:
    Cell data[BOARD_SIZE * BOARD_SIZE];
    int first;
    int last;

public:
    ArrayQueue() {
        first = 0;
        last = 0;
    }

    bool empty() {
        return first == last;
    }

    void push(Cell value) {
        data[last] = value;
        last++;
    }

    Cell pop() {
        Cell value = data[first];
        first++;
        return value;
    }
};

class ListQueue {
private:
    struct Node {
        Cell value;
        Node* next;
    };

    Node* first;
    Node* last;

public:
    ListQueue() {
        first = nullptr;
        last = nullptr;
    }

    ~ListQueue() {
        while (!empty()) {
            pop();
        }
    }

    bool empty() {
        return first == nullptr;
    }

    void push(Cell value) {
        Node* node = new Node;
        node->value = value;
        node->next = nullptr;

        if (last == nullptr) {
            first = node;
            last = node;
        } else {
            last->next = node;
            last = node;
        }
    }

    Cell pop() {
        Node* node = first;
        Cell value = node->value;

        first = first->next;

        if (first == nullptr) {
            last = nullptr;
        }

        delete node;
        return value;
    }
};

bool parseSquare(string text, Cell& cell) {
    if (text.length() != 2) {
        return false;
    }

    char letter = (char)toupper(text[0]);
    char digit = text[1];

    if (letter < 'A' || letter > 'H' || digit < '1' || digit > '8') {
        return false;
    }

    cell.col = letter - 'A';
    cell.row = digit - '1';
    return true;
}

bool insideBoard(int row, int col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

void clearDistances(int distance[BOARD_SIZE][BOARD_SIZE]) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            distance[i][j] = -1;
        }
    }
}

int bfsArrayQueue(Cell start, Cell finish) {
    int distance[BOARD_SIZE][BOARD_SIZE];
    int rowMove[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
    int colMove[8] = {-1, 1, -2, 2, -2, 2, -1, 1};

    clearDistances(distance);

    ArrayQueue q;
    q.push(start);
    distance[start.row][start.col] = 0;

    while (!q.empty()) {
        Cell current = q.pop();

        if (current.row == finish.row && current.col == finish.col) {
            return distance[current.row][current.col];
        }

        for (int i = 0; i < 8; i++) {
            int newRow = current.row + rowMove[i];
            int newCol = current.col + colMove[i];

            if (insideBoard(newRow, newCol) && distance[newRow][newCol] == -1) {
                distance[newRow][newCol] = distance[current.row][current.col] + 1;
                q.push({newRow, newCol});
            }
        }
    }

    return -1;
}

int bfsListQueue(Cell start, Cell finish) {
    int distance[BOARD_SIZE][BOARD_SIZE];
    int rowMove[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
    int colMove[8] = {-1, 1, -2, 2, -2, 2, -1, 1};

    clearDistances(distance);

    ListQueue q;
    q.push(start);
    distance[start.row][start.col] = 0;

    while (!q.empty()) {
        Cell current = q.pop();

        if (current.row == finish.row && current.col == finish.col) {
            return distance[current.row][current.col];
        }

        for (int i = 0; i < 8; i++) {
            int newRow = current.row + rowMove[i];
            int newCol = current.col + colMove[i];

            if (insideBoard(newRow, newCol) && distance[newRow][newCol] == -1) {
                distance[newRow][newCol] = distance[current.row][current.col] + 1;
                q.push({newRow, newCol});
            }
        }
    }

    return -1;
}

int bfsStlQueue(Cell start, Cell finish) {
    int distance[BOARD_SIZE][BOARD_SIZE];
    int rowMove[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
    int colMove[8] = {-1, 1, -2, 2, -2, 2, -1, 1};

    clearDistances(distance);

    queue<Cell> q;
    q.push(start);
    distance[start.row][start.col] = 0;

    while (!q.empty()) {
        Cell current = q.front();
        q.pop();

        if (current.row == finish.row && current.col == finish.col) {
            return distance[current.row][current.col];
        }

        for (int i = 0; i < 8; i++) {
            int newRow = current.row + rowMove[i];
            int newCol = current.col + colMove[i];

            if (insideBoard(newRow, newCol) && distance[newRow][newCol] == -1) {
                distance[newRow][newCol] = distance[current.row][current.col] + 1;
                q.push({newRow, newCol});
            }
        }
    }

    return -1;
}

double measureTime(int (*function)(Cell, Cell), Cell start, Cell finish, int& answer) {
    int sum = 0;

    auto timeStart = chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_COUNT; i++) {
        sum += function(start, finish);
    }

    auto timeFinish = chrono::high_resolution_clock::now();

    answer = sum / TEST_COUNT;
    return chrono::duration<double>(timeFinish - timeStart).count();
}

void printResult(string name, int answer, double seconds) {
    cout << name << ": ";
    cout << "минимум ходов = " << answer << ", ";
    cout << "время теста = " << fixed << setprecision(6) << seconds << " sec\n";
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    cout << "ФИО: Кононенко Г. А.\n";
    cout << "Группа: 090304-РПИа-о25\n\n";

    string firstText;
    string secondText;
    Cell first;
    Cell second;

    cout << "Введите первое поле, например A5: ";
    cin >> firstText;

    cout << "Введите второе поле, например C2: ";
    cin >> secondText;

    if (!parseSquare(firstText, first) || !parseSquare(secondText, second)) {
        cout << "Ошибка: поля должны быть в формате A1, B3, H8.\n";
        cout << "Нажмите Enter для выхода...";
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin.get();
        return 0;
    }

    cout << "\nСтруктура данных: очередь\n";
    cout << "Количество повторов для сравнения времени: " << TEST_COUNT << "\n\n";

    int answerArray;
    int answerList;
    int answerStl;

    double timeArray = measureTime(bfsArrayQueue, first, second, answerArray);
    double timeList = measureTime(bfsListQueue, first, second, answerList);
    double timeStl = measureTime(bfsStlQueue, first, second, answerStl);

    printResult("1. Очередь через массив", answerArray, timeArray);
    printResult("2. Очередь через связанный список", answerList, timeList);
    printResult("3. Очередь через STL queue", answerStl, timeStl);

    cout << "\nНажмите Enter для выхода...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

    return 0;
}
