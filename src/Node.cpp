#include "Node.h"
#include <stdlib.h>
#include <vector>
#include <random>
#include <iostream>

using std::vector;
using std::default_random_engine;
using std:: normal_distribution;
using std::cout;
using std::endl;




Node::Node(int number_weights, int seed,int activation_function)
{
    //ctor
    m_number_weights = number_weights;
    set_random_seed(seed);

    initialize_node(m_number_weights);
    set_activation_function(activation_function);


}

Node::~Node()
{
    //dtor
}


void Node::initialize_node(int number_weights)
{
    m_number_weights = number_weights;
    normal_distribution<double> distribution(0,1.0);

    m_weights.clear();
    for (int i = 0; i < number_weights; i++) {
        m_weights.push_back(distribution(m_generator));
    }

    m_bias = 0.0;
    m_delta = 0.0;

}

void Node::set_all_weight(float value)
{


    for (int i = 0; i < m_number_weights; i++) {
        m_weights[i] = value;
    }

}


void Node::set_weight(int i, float value)
{
    m_weights[i] = value;
}

float Node::get_weight(int i)
{
    return m_weights[i];
}

int Node::get_number_weights()
{
    return m_number_weights;
}

void Node::set_bias(float value)
{
    m_bias = value;
}

float Node::get_bias()
{
    return m_bias;
}

void Node::set_activation_function(int activation_function)
{
    //if 0 -> relu, if 1 -> sigmoid, if 2 -> linear, if -1 -> sign

    m_activation_function = activation_function;

}

int Node::get_activation_function(void)
{
    return m_activation_function;
}

void Node::print_node(void)
{
    cout << "Bias: " << get_bias() << endl;

    for (int i = 0; i < get_number_weights(); i++){


        cout << "Weight " << i << ": " << get_weight(i) << endl;
    }

}

void Node::set_random_seed( int seed)
{
    m_generator.seed (seed);
    m_seed = seed;
}

int Node::get_random_seed( )
{
    return m_seed;
}


void Node::calculate_weighted_sum(vector<float> inputs)
{

    float sum = 0.0;

    if (inputs.size() == m_number_weights){
        for (int i = 0; i < m_number_weights; i++){
            sum += m_weights[i]*inputs[i];
        }

        sum += m_bias;

    }


    m_weighted_sum = sum;
}



void Node::calculate_activation(vector<float> inputs)
{

    float output = 0.0;


    calculate_weighted_sum(inputs);


    if (m_activation_function==0){
        if (m_weighted_sum > 0){
                //relu
            output = 1.0*m_weighted_sum;
        }
    } else if (m_activation_function == 1) {
        //sigmoid
        output = 1/(1+exp(-1*m_weighted_sum));
    } else if (m_activation_function == -1){
        //sign
        output = m_weighted_sum>0;
    }else {
        //linear
        output = 1.0*m_weighted_sum;
    }
    m_activation = output;

    m_activation_derivative = derivative(m_activation);
}


float Node::derivative(float value)
{
    //if 0 -> relu, if 1 -> sigmoid, if 2 -> linear, if -1 -> sign
    float output = 0;
    if (m_activation_function == 0){
        if (value > 0){
            output = 1;
        }
    }
    else if (m_activation_function == 1){
        output = value*(1.0-value);
    }
    else if (m_activation_function == 2){
        output = 1;
    }
    return output;
}


float Node::get_activation_derivative()
{
    return m_activation_derivative;
}

float Node::get_activation()
{
    return m_activation;
}



float Node::get_weighted_sum()
{
    return m_weighted_sum;
}

void Node::update_node(float learning_rate, vector<float> previous_activations)
{

    for (int i = 0; i < m_number_weights; i++){

        m_weights[i] -= learning_rate*m_delta*m_activation_derivative*previous_activations[i];
    }
    m_bias -= learning_rate*m_delta*m_activation_derivative;


}

void Node::set_delta(float value)
{
    m_delta = value;
}

void Node::add_delta(float value)
{
    m_delta += value;
}

float Node::get_delta()
{
    return m_delta;
}


