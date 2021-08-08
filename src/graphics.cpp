
#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <numeric>
#include <unistd.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "ncurses.h"

#include "graphics.h"


void grapher(bool& exit, std::vector<float>& Costs, std::mutex& bufferLock, std::mutex& exitLock)
{
    sf::ContextSettings settings;
    settings.majorVersion = 4;
    settings.minorVersion = 6;
    settings.antialiasingLevel = 8;
 
    sf::RenderWindow window(sf::VideoMode(600,600), "Graph of cost function to epochs", sf::Style::Default, settings);
    sf::RenderWindow();
    
    sf::Vector2f size;

    std::vector<float> localCostsSlice; 
    std::vector<float> cumulativeLocalCosts;
    std::vector<int> xValues;
    
    bool localExit;

    int graphResolution = 100;
    sf::VertexArray plot(sf::LinesStrip, graphResolution+1);

    float frameRate = 60.0; // sets frames per second
    
    float padding = 100; // Padding for graph
    float tickLength = 20; // Length of a tick

    float min, max; // Holds the biggest and smallest values
    
    sf::Font font;
    font.loadFromFile("resources/fonts/Roboto-Thin.ttf");
    sf::Text label;
    label.setFont(font);
    label.setCharacterSize(24);
    label.setFillColor(sf::Color::White);

    unsigned int xTicks = 5; // The amount of ticks which are displayed on the x axis
    unsigned int yTicks = 5; // xTicks but for the y axis

    sf::Vertex xTickArray[xTicks*2];
    sf::Vertex yTickArray[yTicks*2];

    sf::Vertex xGridArray[xTicks*2];
    sf::Vertex yGridArray[yTicks*2];

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(frameRate);

    while(!localExit && window.isOpen()) {
        // Poll events
        sf::Event event;
        while(window.pollEvent(event)) {
            if(event.type==sf::Event::Closed)
                window.close();
        }

        exitLock.lock();
        localExit = exit;
        exitLock.unlock();

        bufferLock.lock();
        localCostsSlice = Costs;
        Costs.resize(0);
        bufferLock.unlock();

        for(int i = 0 ; i < localCostsSlice.size()-1 ; i++) {
            static float w;
            w = *max_element(std::begin(localCostsSlice), std::end(localCostsSlice));
            if(w > max) {
                max = w;
            }
            
            w = *min_element(std::begin(localCostsSlice), std::end(localCostsSlice));
            if(w < min) {
                min = w;
            }
        }

        if (localCostsSlice.size() != 0) {
            cumulativeLocalCosts.insert(
                std::end(cumulativeLocalCosts),
                std::begin(localCostsSlice), std::end(localCostsSlice)
            );

            xValues.resize(cumulativeLocalCosts.size());
            std::iota(std::begin(xValues), std::end(xValues), 0);

            localCostsSlice.resize(0);
        }  

        // GRAPHICS

        size.x = window.getSize().x;
        size.y = window.getSize().y; // Can't cast sf::Vector2u to sf::Vector2f so each component has to be changed individually
        
        sf::Vertex axes[] =
        {
            sf::Vertex(sf::Vector2f(padding,padding), sf::Color::White),
            sf::Vertex(sf::Vector2f(padding,size.y-padding), sf::Color::White),
            sf::Vertex(sf::Vector2f(padding,size.y-padding), sf::Color::White),
            sf::Vertex(sf::Vector2f(size.x-padding,size.y-padding), sf::Color::White)
        };         

        window.clear();

        for(int i = 0 ; i < xTicks ; i++) {
            int currentX = padding + ((size.x-(padding*2)) * ((i+1.0) / xTicks));

            xTickArray[i*2] = sf::Vertex(
                sf::Vector2f(currentX, size.y - padding + (tickLength/2)),
                sf::Color::White
            );
            xTickArray[i*2+1] = sf::Vertex(
                sf::Vector2f(currentX, size.y - padding - (tickLength/2)),
                sf::Color::White
            );

            xGridArray[i*2] = sf::Vertex(
                sf::Vector2f(currentX, padding),
                sf::Color(100,100,100,255)
            );
            
            xGridArray[i*2+1] = sf::Vertex(
                sf::Vector2f(currentX, size.y - padding - (tickLength/2)),
                sf::Color(100,100,100,255)
            );

            label.setString(std::to_string((int)(cumulativeLocalCosts.size()*(i+1.0)/xTicks)));
            label.setPosition(currentX, size.y - padding - (tickLength/2) + 25);
            window.draw(label);
            
            /* Sets a label at each x grid line
            float currentY = size.y - (padding+(size.y-2*padding)*(cumulativeLocalCosts[(int)(cumulativeLocalCosts.size()*(i+1.0)/xTicks)])/cumulativeLocalCosts[0]);

            label.setString(std::to_string(cumulativeLocalCosts[(int)(cumulativeLocalCosts.size()*(i+1.0)/xTicks)]));
            label.setPosition(currentX, currentY);
            window.draw(label);
            */
        }
        
        for(int i = 0 ; i < yTicks + 1 ; i++) {
            int currentY = size.y - (padding + (size.y - padding*2)) * (i+1.0) / yTicks;

            yTickArray[i*2] = sf::Vertex(
                sf::Vector2f(padding + (tickLength/2), currentY),
                sf::Color::White
            );
            yTickArray[i*2+1] = sf::Vertex(
                sf::Vector2f(padding - (tickLength/2), currentY),
                sf::Color::White
            );

            yGridArray[i*2] = sf::Vertex(
                sf::Vector2f(padding, currentY),
                sf::Color(100,100,100,255)
            );
            
            yGridArray[i*2+1] = sf::Vertex(
                sf::Vector2f(size.x - padding, currentY),
                sf::Color(100,100,100,255)
            );

            label.setString(std::to_string(min + (max-min)*(i/(float)yTicks)));
            
            label.setPosition(padding +10, currentY - 40);

            window.draw(label);
        }

        for(int i = 0 ; i < graphResolution ; i++) {
            float mod = (float)i / (float)graphResolution;
            float currentValue = cumulativeLocalCosts[mod * cumulativeLocalCosts.size()];
            plot[i].position = sf::Vector2f(
                padding + (size.x - 2*padding) * mod,
                size.y - (currentValue/max)*(size.y - padding*2)
            );
        }

        plot[graphResolution] = sf::Vertex(
            sf::Vector2f(size.x-padding, plot[graphResolution-1].position.y),
            sf::Color::White
        );

        sf::Vertex pointer[] = {
            sf::Vertex(plot[graphResolution].position, sf::Color(150,150,150,255)),
            sf::Vertex(plot[graphResolution].position - sf::Vector2f(padding,padding), sf::Color(150,150,150,255))
        };

        label.setString(std::to_string(cumulativeLocalCosts[cumulativeLocalCosts.size()-1]));
        label.setPosition(pointer[1].position - sf::Vector2f(30,30));
        window.draw(label);

        label.setString("Evolution of the cost of the functor over time");
        label.setOrigin(label.getLocalBounds().width/2.0f,label.getLocalBounds().height/2.0f);
        label.setPosition(size.x/2, padding/2);
        window.draw(label);

        label.setString("Passes");
        label.setOrigin(label.getLocalBounds().width/2.0f,label.getLocalBounds().height/2.0f);
        label.setPosition(size.x/2, size.y-padding/2);
        window.draw(label);

        label.setString("Cost");
        label.setOrigin(label.getLocalBounds().width/2.0f,label.getLocalBounds().height/2.0f);
        label.setPosition(padding/2, size.y/2);
        label.setRotation(-90);
        window.draw(label);
        label.setRotation(0);

        

        window.draw(xGridArray, xTicks*2, sf::Lines);
        window.draw(yGridArray, yTicks*2, sf::Lines);
        window.draw(axes, 4, sf::Lines);
        window.draw(xTickArray, xTicks*2, sf::Lines);
        window.draw(yTickArray, yTicks*2, sf::Lines);
        window.draw(plot);
        window.draw(pointer, 2, sf::Lines);
            
        window.display();
        // END GRAPHICS
    }      
}