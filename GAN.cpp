#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <random>
#include <assert.h>



int random_int(int min, int max) //range : [min, max]
{
    static bool first = true;
    if (first)
    {
        srand(time(NULL)); //seeding for the first time only!
        first = false;
    }
    return min + rand() % ((max + 1) - min);
}

void tokenize(std::string const& str, const char delim,
    std::vector<std::string>& out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

/*
void get_mnist(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    const std::string inputFile = "C:\\Users\\morit\\Desktop\\CODE\\PYTHON\\Neural_Network\\Build 6\\Build 6.0\\mnist_train.csv";
    std::ifstream inFile(inputFile, std::ios_base::binary);

    inFile.seekg(0, std::ios_base::end);
    size_t length = inFile.tellg();
    inFile.seekg(0, std::ios_base::beg);

    std::vector<char> buffer;
    buffer.reserve(length);
    std::copy(std::istreambuf_iterator<char>(inFile),
        std::istreambuf_iterator<char>(),
        std::back_inserter(buffer));

    /// 28*28 -> 784



    std::string s(buffer.begin(), buffer.end());


    const char delim = '\n';

    std::vector<std::string> out;
    tokenize(s, delim, out);

    //std::vector<std::vector<int>> out2;

    for (int i = 0; i < out.size(); i++) {
        std::vector<std::string> new_vec;
        std::vector<float> true_vec;
        tokenize(out[i], ',', new_vec);

        for (std::string str : new_vec) {
            true_vec.push_back(((float)(atoi(str.c_str()))) / 255.0);
        }

        std::vector<float> label_vec(10, 0);
        label_vec[(int)(true_vec[0] * 255.0f)] = 1;
        labels.push_back(label_vec);
        true_vec.erase(true_vec.begin());
        images.push_back(true_vec);
    }

}*/

void get_mnist(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    const std::string inputFile = "C:\\Users\\morit\\Desktop\\CODE\\PYTHON\\Neural_Network\\Build 6\\Build 6.0\\mnist_train.csv";
    std::ifstream inFile(inputFile, std::ios_base::binary);

    inFile.seekg(0, std::ios_base::end);
    size_t length = inFile.tellg();
    inFile.seekg(0, std::ios_base::beg);

    std::vector<char> buffer;
    buffer.reserve(length);
    std::copy(std::istreambuf_iterator<char>(inFile),
        std::istreambuf_iterator<char>(),
        std::back_inserter(buffer));

    /// 28*28 -> 784



    std::string s(buffer.begin(), buffer.end());


    const char delim = '\n';

    std::vector<std::string> out;
    tokenize(s, delim, out);

    //std::vector<std::vector<int>> out2;

    for (int i = 0; i < out.size(); i++) {
        std::vector<std::string> new_vec;
        std::vector<float> true_vec;
        tokenize(out[i], ',', new_vec);

        for (std::string str : new_vec) {
            true_vec.push_back(((float)(atoi(str.c_str()))) / 255.0);
        }

        std::vector<float> label_vec(10, 0);
        label_vec[true_vec[0] * 255] = 1;
        labels.push_back(label_vec);
        true_vec.erase(true_vec.begin());
        images.push_back(true_vec);
    }

}



void build_rect(std::vector<float>& rect) {
    if (random_int(0, 1)) {
        int x = 5; //random_int(0, 28 - 6);
        int y = 5; //random_int(0, 28 - 6);

        int width = 5;

        for (int _x = 0; _x < width; _x++) {
            for (int _y = 0; _y < width; _y++) {
                rect[(x + _x) * 28 + (y + _y)] = 1;  //x*xMax + y
            }
        }
    } else {
        int x = 10; //random_int(0, 28 - 6);
        int y = 7; //random_int(0, 28 - 6);

        int width = 8;

        for (int _x = 0; _x < width; _x++) {
            for (int _y = 0; _y < width; _y++) {
                rect[(x + _x) * 28 + (y + _y)] = 1;  //x*xMax + y
            }
        }
    }

}


void get_rect_set(std::vector<std::vector<float>>& images, int n) {

    for (int i = 0; i < n; i++) {
        std::vector<float> rect(28 * 28, 0);
        build_rect(rect);
        images.push_back(rect);
    }
}

class NeuralNetwork {

public:
    float e = 2.71828;
    std::vector<int> networkSize;
    float learning_rate = 0.1;
    int number_of_inputs;
    int number_of_layers;
    int number_of_layers_minus_1;
    std::vector<int> layer_neuron_numbers;
    std::vector<std::vector<std::vector<float>>> network;
    std::vector<float> output;

    NeuralNetwork(std::vector<int> _networkSize) {
        networkSize = _networkSize;

        number_of_inputs = networkSize[0];
        number_of_layers = networkSize.size() - 1;
        number_of_layers_minus_1 = number_of_layers - 1;

        std::vector<int> _layer_neuron_numbers(networkSize.begin() + 1, networkSize.end());
        layer_neuron_numbers = _layer_neuron_numbers;

        generate_network();

        //print_network();

    }

    float activation_function(float inp) {
        return 1 / (1 + exp(-inp));
    }

    float d_activation_function(float inp) {
        return inp * (1 - inp);
    }

    void generate_network() {
        int current_number_of_inputs = number_of_inputs;
        int number_of_neurons_in_layer;
        float weight;
        //std::vector<std::vector<float>> layer;
        //std::vector<float> neuron;

        for (int i = 0; i < layer_neuron_numbers.size(); i++) {
            number_of_neurons_in_layer = layer_neuron_numbers[i];
            std::vector<std::vector<float>> layer;

            for (int o = 0; o < number_of_neurons_in_layer; o++) {
                std::vector<float> neuron;

                for (int p = 0; p < (current_number_of_inputs + 1); p++) {
                    weight = rand() / (RAND_MAX + 1.) - 0.5;
                    neuron.push_back(weight);
                }
                layer.push_back(neuron);
            }
            network.push_back(layer);
            current_number_of_inputs = number_of_neurons_in_layer;
        }
    }

    void feed_forward(std::vector<float> inputs) {
        for (std::vector<std::vector<float>> layer : network) {
            std::vector<float> layer_output;

            for (std::vector<float> neuron : layer) {
                float activation = 0;
                for (int i = 0; i < inputs.size(); i++) {
                    //std::cout << neuron[i] << " ";
                    activation += neuron[i] * inputs[i];
                }
                float bias = neuron[neuron.size() - 1];
                activation = activation_function(activation + bias);
                layer_output.push_back(activation);
            }

            inputs = layer_output;
            layer_output.clear();
        }
        output = inputs;
    }

    void train_batch(std::vector<std::vector<float>>& training_inputs, std::vector<std::vector<float>>& targets) {
        std::vector<std::vector<std::vector<float>>> current_network = network;
        for (int training_example_number = 0; training_example_number < training_inputs.size(); training_example_number++) {
            //std::cout << training_inputs.size() << "  " << targets.size();
            std::vector<float> input = training_inputs[training_example_number];
            std::vector<float> target = targets[training_example_number];


            //FORWARD PASS TO GATHER INFO
            std::vector<std::vector<float>> all_layer_outputs;
            std::vector<std::vector<float>> all_layer_inputs;


            for (std::vector<std::vector<float>> layer : network) {
                all_layer_inputs.push_back(input);
                std::vector<float> layer_output;

                for (std::vector<float> neuron : layer) {
                    float activation = 0;
                    for (int i = 0; i < input.size(); i++) {
                        //std::cout << neuron[i] << " ";
                        activation += neuron[i] * input[i];
                    }
                    float bias = neuron[neuron.size() - 1];
                    activation = activation_function(activation + bias);
                    layer_output.push_back(activation);
                }

                input = layer_output;
                all_layer_outputs.push_back(layer_output);
                layer_output.clear();
            }
            output = input;


            // OUTPUT NEURON DELTAS
            std::vector<std::vector<float>> all_neuron_deltas;
            std::vector<float> first_layer_delta;

            for (int i = 0; i < output.size(); i++) {
                float _target = target[i];
                float _output = output[i];

                first_layer_delta.push_back(-(_target - _output) * d_activation_function(_output));
            }
            all_neuron_deltas.push_back(first_layer_delta);

            // HIDDEN DELTAS
            for (int layer_number = 0; layer_number < number_of_layers_minus_1; layer_number++) {
                int real_layer_number = number_of_layers - (layer_number + 2);
                int num_of_neurons_in_layer = layer_neuron_numbers[real_layer_number];
                std::vector<float> layer_deltas;

                std::vector<float> layer_outputs = all_layer_outputs[real_layer_number];

                for (int a = 0; a < num_of_neurons_in_layer; a++) {
                    float neuron_error = 0;
                    int shallower_layer_number = real_layer_number + 1;
                    int num_of_neurons_in_shallower_layer = layer_neuron_numbers[shallower_layer_number];
                    std::vector<std::vector<float>> shallower_layer = network[shallower_layer_number];

                    for (int b = 0; b < num_of_neurons_in_shallower_layer; b++) {
                        neuron_error += shallower_layer[b][a] * all_neuron_deltas[all_neuron_deltas.size() - 1][b];
                    }

                    layer_deltas.push_back(neuron_error * d_activation_function(layer_outputs[a]));
                }

                all_neuron_deltas.push_back(layer_deltas);
            }

            // UPDATE NEURON WEIGHTS
            for (int layer_number = 0; layer_number < number_of_layers; layer_number++) {
                int real_layer_number = number_of_layers_minus_1 - layer_number;
                int num_of_neurons_in_layer = layer_neuron_numbers[real_layer_number];
                //std::vector<float> layer_inputs = all_layer_inputs[real_layer_number];
                for (int neuron_number = 0; neuron_number < num_of_neurons_in_layer; neuron_number++) {
                    int network_access_index = real_layer_number;

                    for (int weight_number = 0; weight_number < networkSize[network_access_index]; weight_number++) {
                        float weight_error = all_neuron_deltas[layer_number][neuron_number] * all_layer_inputs[real_layer_number][weight_number];
                        current_network[real_layer_number][neuron_number][weight_number] -= weight_error * learning_rate;
                    }
                    current_network[real_layer_number][neuron_number].back() -= all_neuron_deltas[layer_number][neuron_number] * learning_rate;
                }
            }
        }
        network = current_network;
    }

    void save_network(std::string name) {
        std::ofstream out(name);
        for (auto& layer : network)
        {
            //std::vector<std::vector<float>>
            for (auto& neuron : layer)
            {
                //std::vector<float>
                for (auto weight : neuron)
                {
                    out << weight << ',';
                }
                out << '\t';
            }
            out << '\n';
        }
    }

    void load_network(const std::string inputFile) { //BIAS BIAS

        std::ifstream inFile("C:\\Users\\morit\\source\\repos\\NeuralNetwork\\NeuralNetwork\\neural_savefile.txt", std::ios_base::binary);

        inFile.seekg(0, std::ios_base::end);
        size_t length = inFile.tellg();
        inFile.seekg(0, std::ios_base::beg);

        std::vector<char> buffer;
        buffer.reserve(length);
        std::copy(std::istreambuf_iterator<char>(inFile),
            std::istreambuf_iterator<char>(),
            std::back_inserter(buffer));

        std::string s(buffer.begin(), buffer.end());

        //tokenize(s, delim, out);
        std::vector<std::string> layers;
        tokenize(s, '\n', layers);


        network.clear();
        //std::vector<std::vector<std::vector<float>>> network;

        std::vector<std::vector<std::string>> neurons;

        for (int i = 0; i < number_of_layers; i++) {

            std::vector<std::vector<float>> network_layer;

            std::string layer = layers[i];

            std::vector<std::string> neurons;
            tokenize(layer, '\t', neurons);

            for (int o = 0; o < layer_neuron_numbers[i]; o++) {

                std::vector<float > network_neuron;

                std::string neuron = neurons[o];

                std::vector<std::string> weights;
                tokenize(neuron, ',', weights);

                for (int p = 0; p < networkSize[i] + 1; p++) {

                    std::string weight = weights[p];

                    network_neuron.push_back(std::atof(weight.c_str()));
                }
                network_layer.push_back(network_neuron);
            }
            network.push_back(network_layer);
        }



    }

    void print_network() {
        for (std::vector<std::vector<float>> layer : network) {
            for (std::vector<float> neuron : layer) {
                for (float weight : neuron) {
                    std::cout << weight << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }
        std::cout << "________________________________________________________________________________________________________________" << std::endl;

    }

};


class GAN {
public:
    std::vector<int> generator_size;
    std::vector<int> discriminator_size;
    std::vector<int> layer_neuron_numbers_of_GAN;
    std::vector<float> random_vector;

    int number_of_Layers_in_GAN;
    int number_of_Layers_minus_1;
    int number_of_Layers_in_generator;
    int number_of_Layers_in_discriminator;
    int number_of_inputs_to_generator;

    float generator_learning_rate = 0.0005;
    float discriminator_learning_rate = 0.0005;
    int generator_training_cycles = 4;

    std::vector<float> v_real = { 0 };
    std::vector<float> v_fake = { 1 };
    int i_real = 0;
    int i_fake = 1;

    NeuralNetwork* generator;
    NeuralNetwork* discriminator;


    GAN(std::vector<int>& _generator_size, std::vector<int>& _discriminator_size) {
        generator_size = _generator_size;
        discriminator_size = _discriminator_size;

        number_of_Layers_in_generator = generator_size.size() - 1;
        number_of_Layers_in_discriminator = discriminator_size.size() - 1;

        number_of_Layers_in_GAN = number_of_Layers_in_generator + number_of_Layers_in_discriminator;
        number_of_Layers_minus_1 = number_of_Layers_in_GAN - 1;

        number_of_inputs_to_generator = generator_size[0];

        for (int i = 1; i <= number_of_Layers_in_generator; i++) {
            layer_neuron_numbers_of_GAN.push_back(generator_size[i]);
        }

        for (int i = 1; i <= number_of_Layers_in_discriminator; i++) {
            layer_neuron_numbers_of_GAN.push_back(discriminator_size[i]);
        }

        for (int i = 0; i < number_of_inputs_to_generator; i++) {
            random_vector.push_back(1.0f);
        }

        //NeuralNetwork _generator(generator_size);
        generator = new NeuralNetwork(generator_size);//&_generator;

        //NeuralNetwork _discriminator(discriminator_size);
        discriminator = new NeuralNetwork(discriminator_size);


        generator->learning_rate = generator_learning_rate;
        discriminator->learning_rate = discriminator_learning_rate;
    }


    ~GAN() {
        delete generator;
        delete discriminator;
    }



    void fill_random_vector(std::vector<float>& v)
    {
        std::srand(unsigned(std::time(nullptr)));
        for (float& el : v) {
            el = (float)rand() / (float)RAND_MAX;
        }
    }

    void create_pic() {
        fill_random_vector(random_vector);
        generator->feed_forward(random_vector);
    }

    void create_pic(std::string file_name) {
        create_pic();

        std::ofstream out(file_name);
        for (auto& num : generator->output)
        {
            out << num << ',';
        }
    }

    std::vector<std::vector<float>>* get_combined_layer_ptr(const int& shallower_layer_number) {
        return (shallower_layer_number < generator->network.size()) ? &(generator->network[shallower_layer_number]) : &(discriminator->network[shallower_layer_number - generator->network.size()]);
    }

    void train_batch(std::vector<std::vector<float>>& training_inputs) {
        assert(training_input.size() == 28 * 28);
        int batch_size = training_inputs.size();

        std::vector<std::vector<float>> discriminator_inputs;
        std::vector<std::vector<float>> discriminator_targets;
        for (int i = 0; i < batch_size; i++) {
            if (std::rand() > (RAND_MAX / 2)) {

                fill_random_vector(random_vector);
                generator->feed_forward(random_vector);
                discriminator_inputs.push_back(generator->output);
                discriminator_targets.push_back(v_fake);

            }
            else {
                discriminator_inputs.push_back(training_inputs[i]);
                discriminator_targets.push_back(v_real);
            }
        }
        discriminator->train_batch(discriminator_inputs, discriminator_targets);

        //std::cout << "after disc training" << "\n";

        for (int q = 0; q < generator_training_cycles; q++) {
            std::vector<std::vector<std::vector<float>>> current_generator_network = generator->network;
            for (int training_example_number = 0; training_example_number < batch_size; training_example_number++) {
                fill_random_vector(random_vector);

                //FORWARD PASS TO GATHER INFO
                std::vector<std::vector<float>> all_layer_outputs;
                std::vector<std::vector<float>> all_layer_inputs;

                std::vector<float> gan_input = random_vector;

                for (std::vector<std::vector<float>> layer : generator->network) {
                    all_layer_inputs.push_back(gan_input);
                    std::vector<float> layer_output;

                    for (std::vector<float> neuron : layer) {
                        float activation = 0;
                        for (int i = 0; i < gan_input.size(); i++) {
                            //std::cout << neuron[i] << " ";
                            activation += neuron[i] * gan_input[i];
                        }
                        float bias = neuron[neuron.size() - 1];
                        activation = generator->activation_function(activation + bias);
                        layer_output.push_back(activation);
                    }

                    gan_input = layer_output;
                    all_layer_outputs.push_back(layer_output);
                    layer_output.clear();
                }

                for (std::vector<std::vector<float>> layer : discriminator->network) {
                    all_layer_inputs.push_back(gan_input);
                    std::vector<float> layer_output;

                    for (std::vector<float> neuron : layer) {
                        float activation = 0;
                        for (int i = 0; i < gan_input.size(); i++) {
                            //std::cout << neuron[i] << " ";
                            activation += neuron[i] * gan_input[i];
                        }
                        float bias = neuron[neuron.size() - 1];
                        activation = generator->activation_function(activation + bias);
                        layer_output.push_back(activation);
                    }

                    gan_input = layer_output;
                    all_layer_outputs.push_back(layer_output);
                    layer_output.clear();
                }



                float disc_output = gan_input[0];

                // OUTPUT NEURON DELTAS
                std::vector<std::vector<float>> all_neuron_deltas;
                std::vector<float> first_layer_delta;

                first_layer_delta.push_back(-(i_real - disc_output) * generator->d_activation_function(disc_output));
                all_neuron_deltas.push_back(first_layer_delta);

                // HIDDEN DELTAS
                for (int layer_number = 0; layer_number < number_of_Layers_minus_1; layer_number++) {
                    int real_layer_number = number_of_Layers_in_GAN - (layer_number + 2);
                    //std::cout << "real_layer_number  " << real_layer_number << std::endl;
                    int num_of_neurons_in_layer = layer_neuron_numbers_of_GAN[real_layer_number];
                    std::vector<float> layer_deltas;

                    std::vector<float> layer_outputs = all_layer_outputs[real_layer_number];

                    for (int a = 0; a < num_of_neurons_in_layer; a++) {
                        float neuron_error = 0;
                        int shallower_layer_number = real_layer_number + 1;
                        int num_of_neurons_in_shallower_layer = layer_neuron_numbers_of_GAN[shallower_layer_number];
                        std::vector<std::vector<float>>& shallower_layer = *get_combined_layer_ptr(shallower_layer_number);

                        for (int b = 0; b < num_of_neurons_in_shallower_layer; b++) {
                            neuron_error += shallower_layer[b][a] * all_neuron_deltas[all_neuron_deltas.size() - 1][b];
                        }

                        layer_deltas.push_back(neuron_error * generator->d_activation_function(layer_outputs[a]));
                    }

                    all_neuron_deltas.push_back(layer_deltas);
                }

                // UPDATE
                for (int layer_number = 0; layer_number < number_of_Layers_in_generator; layer_number++) {
                    int real_layer_number = number_of_Layers_in_generator - layer_number - 1; //!
                    //std::cout << real_layer_number << std::endl;
                    int num_of_neurons_in_layer = generator->layer_neuron_numbers[real_layer_number];
                    int number_of_inputs_to_layer = generator_size[real_layer_number];
                    std::vector<float> layer_inputs = all_layer_inputs[real_layer_number]; //!

                    for (int neuron_number = 0; neuron_number < num_of_neurons_in_layer; neuron_number++) {

                        for (int weight_number = 0; weight_number < number_of_inputs_to_layer; weight_number++) {
                            float weight_error = all_neuron_deltas[layer_number + number_of_Layers_in_discriminator][neuron_number] * layer_inputs[weight_number];
                            float& weight = current_generator_network[real_layer_number][neuron_number][weight_number];
                            weight -= weight_error * generator_learning_rate;

                            assert(!(isinf(weight) || isnan(weight)));

                        }
                        current_generator_network[real_layer_number][neuron_number].back() -= all_neuron_deltas[layer_number + number_of_Layers_in_discriminator][neuron_number] * generator_learning_rate;
                    }

                }

            }
            generator->network = current_generator_network;
        }

        //std::cout << generator->network[0][2][0] << "\n";
        
    }

};

/*
float check_mnist(NeuralNetwork nn, std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    float r = 0;
    for (int i = 0; i < 1000; i++) {
        nn.feed_forward(images[i]);

        if ((std::max_element(labels[i].begin(), labels[i].end()) - labels[i].begin()) == (std::max_element(nn.output.begin(), nn.output.end()) - nn.output.begin())) {
            r++;
        }

    }

    std::cout << r / 1000.0 << "\n";
    return r / 1000.0;
}
*/

int main()
{
    
    std::vector<std::vector<float>> images;
    std::vector<std::vector<float>> labels;

    //get_mnist(images, labels);
    get_rect_set(images, 5000);

    std::vector<int> gen = { 10, 20, 28 * 28 };
    std::vector<int> disc = { 28 * 28, 20, 1 };
    GAN gan(gen, disc);
    //gan.train_batch(images, labels);
    gan.discriminator->learning_rate = 0.0003;
    gan.generator->learning_rate = 0.005;
    //gan.generator->load_network("generator.txt");
    //gan.discriminator->load_network("discriminator.txt");



    std::cout << "loop" << std::endl;
    std::vector<std::vector<float>> _images;
    while (true) {
        for (int i = 1; i < images.size(); i++) {
            _images.push_back(images[i - 1]);

            if (!(i % 50)) {
                gan.train_batch(_images);
                _images.clear();
            }

        }
        gan.fill_random_vector(gan.random_vector);
        gan.generator->feed_forward(gan.random_vector);
        gan.discriminator->feed_forward(gan.generator->output);
        //for(int i = 0; i < gan.discriminator->output[0] * 30; i++)
        //    std::cout << " ";
        //std::cout << "|" << std::endl;
        std::cout << gan.discriminator->output[0] << std::endl;

        if (gan.discriminator->output[0] > 0.85) {
            gan.discriminator->learning_rate *= 0.9;
        }
        else {
            gan.discriminator->learning_rate *= 1.1;
        }


        //std::cout << "Done epoch!" << std::endl;
        gan.create_pic("pic.txt");
    }

    /*
    while (true) {

        for (int i = 0; i < images.size(); i++) {

            gan.train_batch(images[i]);
            //std::cout << i << "\n";
        }

        gan.fill_random_vector(gan.random_vector);
        gan.generator->feed_forward(gan.random_vector);
        gan.discriminator->feed_forward(gan.generator->output);
        //for(int i = 0; i < gan.discriminator->output[0] * 30; i++)
        //    std::cout << " ";
        //std::cout << "|" << std::endl;

        

        gan.create_pic("pic.txt");
    }
    */
    //[50, 100, 28 * 28], [28 * 28, 200, 1]
    
    //gan.generator->save_network("generator.txt");
    //gan.discriminator->save_network("discriminator.txt");


    std::cout << "Done!" << std::endl;
    
}
