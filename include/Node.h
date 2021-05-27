#ifndef NODE_H
#define NODE_H


#include <vector>
#include <random>


using std::vector;
using std::default_random_engine;

class Node
{
    public:
        Node(int number_weights, int seed, int activation);
        virtual ~Node();


        void initialize_node(int number_weights); //builds node with set number of random weights connecting to previous layer

        void set_all_weight(float value);

        void set_weight(int i, float value);
        float get_weight(int i);
        int get_number_weights();


        void set_bias(float value);
        float get_bias();


        void set_activation_function(int activation_function);
        int get_activation_function(void);

        void print_node(void);

        void set_random_seed(int seed);
        int get_random_seed();

        void calculate_activation(vector<float> previous);
        void calculate_weighted_sum(vector<float> previous);

        float derivative(float value);
        float get_activation_derivative();



        float get_activation();
        float get_weighted_sum();

        void update_node(float learning_rate, vector<float> previous_activations);

        void set_delta(float value);
        void add_delta(float value);

        float get_delta();

    protected:

    private:

        float m_weighted_sum;
        float m_activation;
        float m_activation_derivative;
        int m_number_weights;
        vector<float> m_weights;
        int m_activation_function;
        float m_bias;
        float m_delta;
        int m_seed;

        default_random_engine m_generator;

};

#endif // NODE_H
