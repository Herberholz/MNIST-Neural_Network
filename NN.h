//Cody Herberholz
//1/11/2017
//CS 445
//HW 1

//When reading in training examples, will read in each line of the file to the 
//input array and use it to train the Neural Network. After training for one Epoch
//is complete then the accuracy will be tested for both the training and test data.
//From here it will load a new training example and repeat the cycle.


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cmath>

using namespace std;

#define MOMENTUM   0.9
#define HIDDENSIZE 100


//Represents a single Perceptron, contains the weights and activation value along
//with the functions to fill these values.
class Neuron
{
    public:
        Neuron();
        ~Neuron();
        void connect(Neuron toConnect[], int layerSize);
        void setWeight(int);
        void findActivation();
        void findActivation(double []); //calculates sum of all weights * inputs
        double getActivation();
        void outputError(double);
        void hiddenError(int);
        double sigmoid(double);
        double sigmoidDeriv();
        void updateWeights(double [], double, int); //uses perceptron learning rule

    private:
        Neuron ** adj;
        double * weight;     //holds all weights connected to inputs
        double * prevChange; //holds the amount of change between weight updates
        double activation;   //holds the sum of all weights * inputs for the output
        double error;        //error percentage given by distance from target
};



//Controls the flow of the Neural Network by taking care of file usage, training,
//and testing of accuracy.
class DigitRecognizer
{
    public: 
        DigitRecognizer();
        void feedForward();
        void backProp();
        int findLargest(); //finds largest sum  which is the prediction
        void updateWeights(double []); //determines which output perceptrons to update
        void load(ifstream &); //loads examples from a file
        void train(); //goes through all training examples in file and updates weights 
        void makeMatrix(int confMatrix [][10]); //produces confusion matrix
        void testAccuracy(); //tests accuracy of NN with both training/test data

    private:
        Neuron hidden[HIDDENSIZE];
        Neuron output[10]; //array of 10 output perceptrons
        double input[785]; //array of 784 inputs and 1 bias
        int target[10];    //array of 10 that depicts which output is the target
        double learningRate; //holds the learning rate for the perceptron learning rule
};
