#include "test_functions.h"
#include "Node.h"
#include "Network.h"
#include "test_functions.h"


using std::cout;
using std::endl;



//Function to test if a single node is correctly calculating the weights sum and activation.
void test_node()
{
    vector<float> inputs = {-0.23423,-1,2};

    Node node (inputs.size(),0,1);

    node.set_all_weight(1.0);
    node.set_bias(-1.0);
    node.set_delta(1.0);
    cout << "Network Sturcture" << endl;
    node.print_node();

    node.calculate_activation(inputs);



    cout << "\n\nExpected Values" << endl;
    cout << "Activation: 0.4417\tWeighted Sum: -0.23423\nActivation Derivative: 0.2466 " << endl;

    cout << "\nCalculated Values" << endl;
    cout << "Activation: " << node.get_activation() << "\tWeighted Sum: " << node.get_weighted_sum() << "\nActivation Derivative: " << node.get_activation_derivative()<< endl;
}


//function to make running logic gate networks easier. Takes a network, two inputs and the target value.
void run_network(Network network, int input_1, int input_2, int target)
{
    vector<float> inputs = {input_1,input_2};

    network.calculate(inputs);
    vector<vector<float>> outputs = network.get_activations();

   // vector<vector<float>> outputs = network.calculate(inputs);

    cout << "Target: " << target << "\tOutput: " << outputs[outputs.size()-1][0] << "\n" <<  endl;

}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions to run neural netowrks with weights to do the basic logic gates. Assumes sign activation function
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void test_and_network()
{

    vector<int> structure = {1};
    int previous;

    Network network(2,structure,-1);

    network.set_weight(0,0,0,1.0);
    network.set_weight(0,0,1,1.0);
    network.set_bias(0,0,-1.5);
    network.print_network();
    cout << "\n\n" << endl;
    run_network(network,0,0,0);
    run_network(network,0,1,0);
    run_network(network,1,0,0);
    run_network(network,1,1,1);

}

void test_or_network()
{

    vector<int> structure = {1};
    int previous;

    Network network(2,structure,-1);

    network.set_weight(0,0,0,1.0);
    network.set_weight(0,0,1,1.0);
    network.set_bias(0,0,-0.5);
    network.print_network();

    run_network(network,0,0,0);
    run_network(network,0,1,1);
    run_network(network,1,0,1);
    run_network(network,1,1,1);

}

void test_nand_network()
{

    vector<int> structure = {1};
    int previous;

    Network network(2,structure,-1);

    network.set_weight(0,0,0,-1.0);
    network.set_weight(0,0,1,-1.0);
    network.set_bias(0,0,2.0);
    network.print_network();

    run_network(network,0,0,1);
    run_network(network,0,1,1);
    run_network(network,1,0,1);
    run_network(network,1,1,0);

}


void test_nor_network()
{

    vector<int> structure = {1};
    int previous;

    Network network(2,structure,-1);

    network.set_weight(0,0,0,-1.0);
    network.set_weight(0,0,1,-1.0);
    network.set_bias(0,0,1.0);
    network.print_network();

    run_network(network,0,0,1);
    run_network(network,0,1,0);
    run_network(network,1,0,0);
    run_network(network,1,1,0);

}

void test_xor_network()
{

    vector<int> structure = {2,1};
    int previous;

    Network network(2,structure,-1);
    //OR gate
    network.set_weight(0,0,0,1.0);
    network.set_weight(0,0,1,1.0);
    network.set_bias(0,0,-0.5);

    //NAND gate
    network.set_weight(0,1,0,-1.0);
    network.set_weight(0,1,1,-1.0);
    network.set_bias(0,1,1.5);

    //AND gate
    network.set_weight(1,0,0,1.0);
    network.set_weight(1,0,1,1.0);
    network.set_bias(1,0,-1.5);

    network.print_network();

    run_network(network,0,0,0);
    run_network(network,0,1,1);
    run_network(network,1,0,1);
    run_network(network,1,1,0);

}


//testing the calculation of the derivative of the loss in respect to a nodes activation. Used to upadte weights.
void test_delta_calc()
{
    vector<float> target = {10.0};
    vector<int> structure = {2,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);
    network.set_all_bias(0.0);
    network.set_all_weights(1.0);

    network.print_network();

    vector<float> inputs = {0,1};

    float err = 0.0;
    cout << "Number Layers: " << number_layers << endl;
    network.calculate(inputs);
    vector<vector<float>> outputs = network.get_activations();
    cout << "Outputs Calculated" << endl;
    network.update_delta(outputs,target,1);
    vector<vector<float>> deltas = network.get_deltas();
    for (int i = deltas.size()-1; i >= 0; i--) {
        for (int j = 0; j < deltas[i].size(); j++){
            cout << "i: " << i << "\tj: " << j << "\tDelta: " << deltas[i][j] << endl;

        }

    }
    cout << "Again" << endl;
    network.update_delta(outputs,target,1);
    deltas = network.get_deltas();
    for (int i = deltas.size()-1; i >= 0; i--) {
        for (int j = 0; j < deltas[i].size(); j++){
            cout << "i: " << i << "\tj: " << j << "\tDelta: " << deltas[i][j] << endl;

        }

    }
    cout << "Reset" << endl;
    network.update_delta(outputs,target,0);
    deltas = network.get_deltas();
    for (int i = deltas.size()-1; i >= 0; i--) {
        for (int j = 0; j < deltas[i].size(); j++){
            cout << "i: " << i << "\tj: " << j << "\tDelta: " << deltas[i][j] << endl;

        }

    }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions to train neural netowrks to learn each of the logic gates.
// This is purely a test to see if learning is possible. Due to running the same data multiple times this is effectively memorizing the transformation.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void test_and_update()
{
    vector<int> structure = {10,10,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{0},{0},{0},{1}};

    float err = 0.0;
    for (int i = 0; i < 500000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.001,inputs);
        }
    }
    cout << "\n\nAND Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }
}

void test_or_update()
{
    vector<int> structure = {10,10,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{0},{1},{1},{1}};

    float err = 0.0;
    for (int i = 0; i < 500000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.001,inputs);
        }
    }
    cout << "\n\nOR Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }
}

void test_nor_update()
{
    vector<int> structure = {10,10,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{1},{0},{0},{0}};

    float err = 0.0;
    for (int i = 0; i < 500000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.001,inputs);
        }
    }
    cout << "\n\nNOR Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }
}

void test_nand_update()
{
    vector<int> structure = {10,10,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{1},{1},{1},{0}};

    float err = 0.0;
    for (int i = 0; i < 500000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.001,inputs);
        }
    }
    cout << "\n\nNAND Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }
}

void test_xor_update()
{

    vector<int> structure = {10,10,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{0},{1},{1},{0}};

    float err = 0.0;
    for (int i = 0; i < 500000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.001,inputs);
        }
    }
    cout << "\n\nXOR Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }

}


void test_xor_small_update()
{

    vector<int> structure = {2,1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{0},{1},{1},{0}};

    float err = 0.0;
    for (int i = 0; i < 1000000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.001,inputs);
        }
    }
    cout << "\n\nXOR Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }
    network.print_network();
}


void test_nand_small_update()
{
   // vector<float> target = {0.9};
    vector<int> structure = {1};
    int number_layers = structure.size();
    float delta;

    int previous;

    Network network(2,structure,1);



    vector<vector<float>> inputs_vec = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<float>> targets_vec = {{1},{1},{1},{0}};

    float err = 0.0;
    for (int i = 0; i < 10000; i++){
        for (int j = 0; j < inputs_vec.size(); j++){
            vector<float> inputs = inputs_vec[j];
            vector<float> target = targets_vec[j];

            network.calculate(inputs);
            vector<vector<float>> outputs = network.get_activations();

            network.update_delta(outputs,target,0);
            network.update_nodes(0.01,inputs);
        }
    }
    cout << "\n\nNAND Gate" << endl;
    for (int j = 0; j < inputs_vec.size(); j++){
        vector<float> inputs = inputs_vec[j];
        vector<float> target = targets_vec[j];

        network.calculate(inputs);
        vector<vector<float>> outputs = network.get_activations();
        cout << "Inputs: " << inputs[0] << ", " << inputs[1] << "\tTargets: " << target[0] << "\tOutput: " << outputs[outputs.size()-1][0] << endl;
    }
}
