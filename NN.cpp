//Cody Herberholz
//1/11/2017
//CS445
//HW1

#include "NN.h"


//Neuron Constructor
Neuron::Neuron()
{
    adj = NULL;
    weight = NULL;
    prevChange = NULL;
    activation = 0.0; //initalizes activation value 
    error = 0.0;
}



//Neuron Deconstructor
Neuron::~Neuron()
{
    delete [] adj;
    delete [] weight;
    delete [] prevChange;
    adj = NULL;
    weight = NULL;
}


//Task: Connects the hidden and output layers
void Neuron::connect(Neuron toConnect[], int layerSize)
{
    adj = new Neuron * [layerSize];

    for(int i = 0; i < layerSize; ++i)
        adj[i] = &toConnect[i];

    return;
}



//Task: Sets weights
void Neuron::setWeight(int layerSize)
{
    weight = new double [layerSize];
    prevChange = new double [layerSize];

    //assign random weights
    for(int i = 0; i < layerSize; ++i)
    {
        weight[i] = (((double)rand() / (double)RAND_MAX) * 0.1) - 0.05;
        prevChange[i] = 0.0;
    }
    
    return;
}



//Task: Calculates activation for output
void Neuron::findActivation()
{
    double sum = 0.0;

    //weight 0 - HIDDENSIZE-1 connects
    //hidden 0 - HIDDENSIZE-1
    for(int i = 0; i < HIDDENSIZE; ++i)
        sum += weight[i] * adj[i]->activation;
    
    //used last weight in array to become bias
    sum += weight[HIDDENSIZE];

    activation = sigmoid(sum);

    return;
}



//Task:   Sums up the Neurons weights * inputs
//Input:  The input array, filled with features from training/test data
//Output: The sum is output from the perceptron in order to be stored
void Neuron::findActivation(double input[])
{
    double sum = 0.0; //holds the sum of weights * inputs

    //input[0] is 1 so weight[0] is bias
    for(int i = 0; i < 785; ++i)
        sum += weight[i] * input[i];

    activation = sigmoid(sum);

    return;
}



//Task: Returns activation
double Neuron::getActivation()
{
    return activation;
}



//Task: Calculates output error
void Neuron::outputError(double targetVal)
{
    double val = 0.0;

    //if target then val becomes 0.9, otherwise 0.1
    if(targetVal)
        val = 0.9;
    else
        val = 0.1;

    //cout << val << endl;
    error = (val - activation) * sigmoidDeriv();

    return;
}



//Task: Calculates hidden error
void Neuron::hiddenError(int hiddenLocation)
{
    double tempError = 0.0;

    for(int i = 0; i < 10; ++i)
        tempError += (adj[i]->error * adj[i]->weight[hiddenLocation]);

    tempError *= sigmoidDeriv();
    error = tempError;

    return;
}


//Task: Sigmoid function
double Neuron::sigmoid(double input)
{
    return (1.0 / (1.0 + exp(-input)));
}



//Task: Derivative of the sigmoid function
double Neuron::sigmoidDeriv()
{
    return (activation * (1.0 - activation));
}



//Task:   Updates the weights if the target was not the same as the output
//Input:  Takes in the target, the input array, learning rate, and highest value which
//        says if the current output is the predicted output
//Output: The weights are changed to try to counteract the error
void Neuron::updateWeights(double input[], double learningRate, int weightSize)
{
    double weightChange = 0.0;

    //make sure to update weight[20] for output weight so if layerSize = 10
    //update output weights
    if(weightSize == HIDDENSIZE)
    {
        for(int i = 0; i < weightSize; ++i)
        {
            weightChange = (learningRate * error * adj[i]->activation) + (MOMENTUM * prevChange[i]);
            weight[i] += weightChange;
            prevChange[i] = weightChange;
        }
        //update bias weight
        weightChange = (learningRate * error * 1.0) + (MOMENTUM * prevChange[HIDDENSIZE]);
        weight[HIDDENSIZE] += weightChange;
        prevChange[HIDDENSIZE] = weightChange;
    }
    else
    {
        for(int i = 0; i < weightSize; ++i)
        {
            weightChange = (learningRate * error * input[i]) + (MOMENTUM * prevChange[i]);
            weight[i] += weightChange;
            prevChange[i] = weightChange;
        }
    }
    return;
}



//DigitRecognizer Constructor
DigitRecognizer::DigitRecognizer()
{
    learningRate = 0.1; //signifies how much change the weight undergoes

    //connects adj pointers to Vertices and randomly sets weights 
    for(int i = 0; i < HIDDENSIZE; ++i)
    {
        hidden[i].connect(output, 10);
        hidden[i].setWeight(785);
        if(i < 10)
        {
            output[i].connect(hidden, HIDDENSIZE);
            output[i].setWeight(HIDDENSIZE + 1); //changed to include bias weight
        }
    }
}



//Task: Calculates all the activations
void DigitRecognizer::feedForward()
{
    //gets activation for hidden Neurons
    for(int i = 0; i < HIDDENSIZE; ++i)
        hidden[i].findActivation(input);

    //gets activation for output Neurons
    for(int i = 0; i < 10; ++i)
        output[i].findActivation();

    return;
}



//Task: Calculates errors and updates weights
void DigitRecognizer::backProp()
{
    for(int i = 0; i < 10; ++i)
        output[i].outputError(target[i]);
    for(int i = 0; i < HIDDENSIZE; ++i)
        hidden[i].hiddenError(i);
    for(int i = 0; i < 10; ++i)
    {
        if(i < 10)
            output[i].updateWeights(input, learningRate, HIDDENSIZE);
        
        hidden[i].updateWeights(input, learningRate, 785);
    }
}



//Task:   Finds the largest sum out of all the output perceptrons and says that is the prediction
//Input:  Takes in an array of sums in order to find the largest
//Output: Outputs the index which indicates which output is the prediction
int DigitRecognizer::findLargest()
{
    double winner = -50000.0; //holds the largest sum, initially negative in order to deal with negative sums
    double largest = 0;
    int index = -1; //holds the index which indicates which output is the prediction

    for(int i = 0; i < 10; ++i)
    {
        largest = output[i].getActivation();
        if(winner < largest)
        {
            winner = largest;
            index = i;
        }
    }

    return index;
}



//Task:   Reads in one training example at a time from the designated file. The 
//        first value is used to determine the target value. Index 0 of input holds
//        the bias which is followed by the features of the training example, indicated
//        by pixels that have been divided by 225 to prevent weights from growing too large
//Input:  Takes in the ifstream file value that is connected to the appropriate file
//Output: Loads values into the target array and the input array
void DigitRecognizer::load(ifstream & read)
{
    int tempTarget = 0.0; //holds the training example value

    read >> tempTarget; //stores actual number
    read.ignore(); //ignores comma

    if(!read) //if end of file then leave function
        return;

    //sets the array to indicate which index is the target
    for(int i = 0; i < 10; ++i)
    {
        if(i == tempTarget)
            target[i] = 1;
        else
            target[i] = 0;
    }

    input[0] = 1; //sets bias to 1

    // bias took up position 0 so start at 1 and fill in each pixel greyscale num
    for(int i = 1; i < 785; ++i)
    {
        read >> input[i];
        read.ignore();

        input[i] = input[i] / 255.0; //divide by largest greyscale value to keep input a fraction
    }

    return;
}



//Task:   Uses training set data to train the perceptrons. Each cycle consists of reading 
//        in one number from the training data, then updateWeights sees if predication lines
//        up with target. If not then it updates weights
//Input:  Uses the class input array value and takes in training data from the file
//Output: After training is complete the weights of each output is sucessfully changed
void DigitRecognizer::train()
{
    ifstream read; 
//    double sums[10]; //holds all output sums in an array

    read.open("mnist_train.csv");
    load(read); //loads one training example

    while(read)
    {
        feedForward();
        backProp();
        load(read); //load new training example
    }
    read.close();
    read.clear();

    return;
}



//Task:   Constructs a confusion matrix that determines what prediction numbers are
//        given for target numbers
//Input:  Takes in an empty matrix in order to fill
//Output: Outputs a complete confusion matrix
void DigitRecognizer::makeMatrix(int confMatrix [][10])
{
    ifstream read;
    int index = -1;  //used to determine prediction array index

    read.open("mnist_test.csv"); //opens test data file
    load(read); //loads one test example

    while(read)
    {
        feedForward();
        index = findLargest();

        //uses target to determine column and then index to determine row
        for(int i = 0; i < 10; ++i)
        {
            //adds 1 to the matrix using index as the row and target as the column
            if(target[i])
                confMatrix[i][index] += 1;
        }
        load(read);
    }
    read.close();
    read.clear();

    return;
}



//Task:   Determines accuracy of the Neural Network using both training and test data
//Input:  The training/test examples from the appropriate files
//Output: Displays how many correct predictions there were compared to total examples
void DigitRecognizer::testAccuracy()
{
    ifstream read;
    int reps = 0;     //counts how many training examples are gone through in a file
    int accuracy = 0; //counts how many correct predictions were made
    int index = -1;   //used to determine prediction array index

    //Tests accuracy of training data and then test data
    for(int j = 0; j < 2; ++j)
    {
        accuracy = 0; //resets accuracy 
        reps = 0;     //resets reps

        //used to find training accuracy first and then test
        if(!j)
            read.open("mnist_train.csv");
        else
            read.open("mnist_test.csv");

        load(read); //takes in one example from file

        while(read)
        {
            feedForward();
            index = findLargest();

            if(target[index]) //if prediction matches target increase accuracy
                ++accuracy;
            
            load(read); //take in new example from file
            ++reps;
        }
        //displays accuracy of either training or test data
        if(!j)
            cout << "Training Accuracy: " << accuracy << "/" << reps << endl; 
        else
            cout << "Test Accuracy:     " << accuracy << "/" << reps << endl;

        read.close();
        read.clear();
    }
    return;
}

