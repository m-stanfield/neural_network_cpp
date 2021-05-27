#include "Network.h"
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


Network::Network(int input_shape, vector<int> nodes_per_layer, int activation_function)
{
    m_input_shape = input_shape;
    m_number_layers = nodes_per_layer.size();
    m_nodes_per_layer = nodes_per_layer;
    m_activation_function = activation_function;
    m_loss_number = 0;
    build();
}

Network::~Network()
{
    //dtor
}


void Network::set_structure(vector<int> nodes_per_layer)
{
    m_nodes_per_layer = nodes_per_layer;

}

int Network::get_number_layers()
{
    return m_number_layers;
}

int Network::get_number_nodes(int layer_number)
{
    return m_network[layer_number].size();
}

vector<int> Network::get_structure()
{
    return m_nodes_per_layer;
}
void Network::calculate(vector<float> input)
{
    vector<vector<float>> activation;

    for (int i = 0; i < m_number_layers; i ++){
        vector<float> layer(m_nodes_per_layer[i]);
        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            if (i == 0){
               m_network[i][j].calculate_activation(input);
            }
            else{
                m_network[i][j].calculate_activation(activation[i-1]);

            }
            layer[j] = m_network[i][j].get_activation();
        }
        activation.push_back(layer);
    }
    m_activations = activation;

}

vector<vector<float>> Network::get_activations()
{
    return m_activations;
}

Node Network::get_node(int layer_number, int node_number)
{
    return m_network[layer_number][node_number];
}

void Network::set_input_shape(int input_shape)
{
    m_input_shape = input_shape;
}

int Network::get_input_shape()
{
    return m_input_shape;
}

void Network::build()
{
    vector<Node> layer;
    m_network.clear();
    int counter = 0;
    for (int i = 0; i < m_nodes_per_layer.size(); i++){
        layer.clear();

        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            if (i == 0){
                Node node(m_input_shape, m_seed+counter, m_activation_function);
                node.set_delta(0.0);
                layer.push_back(node);
            }
            else{
                Node node(m_nodes_per_layer[i-1],m_seed+counter,m_activation_function);
                node.set_delta(0.0);
                layer.push_back(node);
            }


            counter++;
       //     cout << "i: " << i << "\tj: "<<  j << endl;
        }
        m_network.push_back(layer);
    }
    cout << "~~~~~Network Built~~~~~" << endl;
}
void Network::print_network()
{
    for (int i = 0; i < m_nodes_per_layer.size(); i++){
        for (int j = 0; j < m_nodes_per_layer[i]; j ++){
            cout << "\nNode " << i << "\t" << j << endl;
            print_node(i,j);
        }
    }

}

void Network::print_node(int layer_number, int node_number)
{
    Node node = m_network[layer_number][node_number];
    node.print_node();
}

void Network::set_activation_function(int layer_number, int node_number, int activation_function)
{
    m_network[layer_number][node_number].set_activation_function(activation_function);
}

int Network::get_activation_function(int layer_number, int node_number)
{
    return m_network[layer_number][node_number].get_activation_function();
}

void Network::set_loss(int loss_number)
{
    m_loss_number = loss_number;
}

int Network::get_loss()
{
    return m_loss_number;
}

void Network::set_weight(int layer_number, int node_number, int weight_number, float value)
{
    m_network[layer_number][node_number].set_weight(weight_number,value);
}

void Network::set_bias(int layer_number, int node_number, float value)
{
    m_network[layer_number][node_number].set_bias(value);

}

void Network::set_all_weights(float value)
{
    for (int i = 0; i < m_number_layers; i++){
        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            m_network[i][j].set_all_weight(value);
        }
    }
}

void Network::set_all_bias(float value)
{
   for (int i = 0; i < m_number_layers; i++){
        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            m_network[i][j].set_bias(value);
        }
    }
}



float Network::get_weight(int layer_number, int node_number, int weight_number)
{
    return m_network[layer_number][node_number].get_weight(weight_number);

}

float Network::get_bias(int layer_number, int node_number)
{
    return m_network[layer_number][node_number].get_bias();

}

void Network::zero_delta()
{
    for (int i = 0; i < m_number_layers; i++){
        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            m_network[i][j].set_delta(0.0);
        }
    }
}
void Network::update_delta(vector<vector<float>> output, vector<float> targets, int add_old)
{
    float delta;
    float sum;
    float activation_derivative;
    for (int i = (m_number_layers -1); i >= 0; i--){
        delta = 0.0;
        for (int j = 0; j < m_nodes_per_layer[i]; j++){

            if (i == (m_number_layers-1)){
                delta = output[i][j] - targets[j];

            }
            else{
                for (int k = 0; k < m_nodes_per_layer[i+1]; k++){
                    activation_derivative = m_network[i+1][k].get_activation_derivative();
                    delta += m_network[i+1][k].get_delta()*activation_derivative*m_network[i+1][k].get_weight(j);
                }
            }
            if (add_old == 0){
                m_network[i][j].set_delta(delta);
            }
            else{
                m_network[i][j].add_delta(delta);
            }
        }
    }
}

vector<vector<float>> Network::get_deltas()
{
    vector<vector<float>> deltas;
    vector<float> layer_deltas;
    for (int i = 0; i < m_number_layers; i ++){
        layer_deltas.clear();
        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            layer_deltas.push_back(m_network[i][j].get_delta());
        }
        deltas.push_back(layer_deltas);
    }
    return deltas;
}



void Network::update_nodes(float learning_rate, vector<float> inputs)
{
    for (int i = (m_number_layers -1); i >= 0; i--){
        for (int j = 0; j < m_nodes_per_layer[i]; j++){
            if (i == 0){
                m_network[i][j].update_node(learning_rate,inputs);

            }
            else{
                m_network[i][j].update_node(learning_rate,m_activations[i-1]);

            }
        }
    }
}
