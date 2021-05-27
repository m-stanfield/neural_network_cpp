#ifndef NETWORK_H
#define NETWORK_H


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

class Network
{
    public:
        Network(int input_shape, vector<int> nodes_per_layer, int activation_function);
        virtual ~Network();

        void set_structure(vector<int> nodes_per_layer);
        vector<int> get_structure();

        int get_number_layers();
        int get_number_nodes(int layer_number);

        void calculate(vector<float> input);
        vector<vector<float>> get_activations();

        Node get_node(int layer_number, int node_number);

        void set_input_shape(int input_shape);
        int get_input_shape();

        void build();
        void print_network();
        void print_node(int layer_number, int node_number);

        void set_activation_function(int layer_number, int node_number, int activation_function);
        int get_activation_function(int layer_number, int node_number);

        void set_loss(int loss_number);
        int get_loss();

        void set_weight(int layer_number, int node_number, int weight_number, float value);
        void set_bias(int layer_number, int node_number, float value);

        void set_all_weights(float value);
        void set_all_bias(float value);

        float get_weight(int layer_number, int node_number, int weight_number);
        float get_bias(int layer_number, int node_number);

        void zero_delta();
        void update_delta(vector<vector<float>> output, vector<float> targets, int add_old);

        vector<vector<float>> get_deltas();

        void update_nodes(float learning_rate,vector<float> inputs);

    protected:

    private:
        vector<vector<Node>> m_network;
        vector<vector<float>> m_activations;
        int m_number_layers;
        vector<int> m_nodes_per_layer;
        int m_seed;
        int m_activation_function;
        int m_input_shape;
        int m_loss;
        int m_loss_number;
};

#endif // NETWORK_H
