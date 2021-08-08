#include <iostream>

void ping() {
    static int count = 0;
    std::cout << "Ping " << ++count << std::endl;
}

void ping(const char* st) {
    static int count = 0;
    std::cout << st << " " << ++count << std::endl;
}