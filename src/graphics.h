#ifndef _ROOT_DRAW_H
#define _ROOT_DRAW_H

#include <iostream>
#include <vector>
#include <mutex>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

void grapher(bool& exit, std::vector<float>& Costs, std::mutex& lock, std::mutex& exitLock);

#endif