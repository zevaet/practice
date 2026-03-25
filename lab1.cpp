#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <cstdlib>

using namespace std;

int main() {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    cout << "ФИО: Кононенко Г. А." << endl;
    cout << "Группа: 090304-РПИа-о25" << endl << endl;

    int N;
    cout << "Введите N: ";
    cin >> N;

    if (N <= 0) {
        cout << "N должно быть натуральным числом." << endl;
        system("pause");
        return 0;
    }

    if (N % 2 == 0 || N % 5 == 0) {
        cout << "NO" << endl;
        system("pause");
        return 0;
    }

    vector<bool> used(N, false);
    int remainder = 1 % N;
    string result = "1";

    while (!used[remainder]) {
        if (remainder == 0) {
            cout << "Ответ: " << result << endl;
            system("pause");
            return 0;
        }

        used[remainder] = true;
        remainder = (remainder * 10 + 1) % N;
        result += '1';
    }

    cout << "NO" << endl;
    system("pause");
    return 0;
}