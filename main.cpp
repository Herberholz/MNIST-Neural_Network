//Cody Herberholz
//1/11/2017
//CS 445
//HW1

#include "NN.h"



int main()
{
    srand(time(NULL)); //sets a random seed in order to randomize weights
    
    DigitRecognizer controller; //instance of digit recognizer
    int epoch = 0; //holds how many epochs have passed
    int confMatrix[10][10]; //holds the confusion matrix

    cout << "\nTraining Data" << endl;

    cout << "Epoch: " << epoch << endl;
    controller.testAccuracy(); //tests accuracy without updating initial weights
    cout << endl;

    do
    {
        ++epoch;
        cout << "Epoch: " << epoch << endl;
        controller.train(); //updates weights of outputs
        controller.testAccuracy(); //tests accuracy without updating weights
    }while(epoch < 50);

    //fills confusion matrix
    controller.makeMatrix(confMatrix);

    //displays confusion matrix
    cout << "-----Confusion Matrix-----" << endl;

    for(int i = 0; i < 10; ++i)
    {
        for(int j = 0; j < 10; ++j)
        {
            cout << setw(5) <<  confMatrix[j][i];
        }
        cout << endl;
    }

    return 0;
}
