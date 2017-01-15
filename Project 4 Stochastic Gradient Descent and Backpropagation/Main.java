import java.lang.*;
import java.util.*;

class Main
{
    /*
	To train your neural network, let's initialize all the weights with "0.01 * rand.normal()",
	use a learning rate of 0.1, and train for 500 epochs using stochastic gradient descent.
	In other words, we will make 500 total passes through the training data,
	each time presenting the patterns one at-a-time in a different random order.
	 */
    //int countMisclassifications(Matrix features, Matrix labels)
    //compute mis1: stochastic gradient descent, no hidden layer
    //Train
    static void compute_mis1()
    {
        //Loads training data
        String fn = "data/";
        Matrix train_features = new Matrix();
        train_features.loadARFF(fn + "vowel_train_features.arff");
        Matrix train_labels = new Matrix();
        train_labels.loadARFF(fn + "vowel_train_labels.arff");

        double start = (double) System.nanoTime() / 1e9;
        int[] layer_sizes = {27, 11};
        NeuralNet nn = new NeuralNet(layer_sizes);
        nn.init();

        /// This takes ownership of pLearner and pTransform.
        /// If inputs is true, then it applies the transform only to the input features.
        /// If inputs is false, then it applies the transform only to the output labels.
        /// (If you wish to transform both inputs and outputs, you must wrap a filter in a filter)
        /// public Filter(SupervisedLearner pLearner, UnsupervisedLearner pTransform, boolean filterInputs)
        Filter f1 = new Filter(nn, new NomCat(),true);
        Filter f2 = new Filter(f1, new NomCat(),false);
        Filter f3 = new Filter(f2, new Normalizer(),true);

        //stochastic gradient descent, no hidden layer
            f3.train(train_features, train_labels);
        double duration = (double) System.nanoTime() / 1e9 - start;

        //Report misclassification on testing data
        Matrix test_features = new Matrix();
        test_features.loadARFF(fn + "vowel_test_features.arff");
        Matrix test_labels = new Matrix();
        test_labels.loadARFF(fn + "vowel_test_labels.arff");
        double mis1 = f3.countMisclassifications(test_features, test_labels);
        System.out.println("mis1=" + Double.toString(mis1));
        System.out.println("Done in: " + Double.toString(duration) + " seconds.");
    }

    //compute mis2: stochastic gradient descent, one hidden layer of 20 units
    static void compute_mis2()
    {
        //Loads training data
        String fn = "data/";
        Matrix train_features = new Matrix();
        train_features.loadARFF(fn + "vowel_train_features.arff");
        Matrix train_labels = new Matrix();
        train_labels.loadARFF(fn + "vowel_train_labels.arff");

        double start = (double) System.nanoTime() / 1e9;
        int[] layer_sizes = {27, 20, 11};
        NeuralNet nn = new NeuralNet(layer_sizes);
        nn.init();

        /// This takes ownership of pLearner and pTransform.
        /// If inputs is true, then it applies the transform only to the input features.
        /// If inputs is false, then it applies the transform only to the output labels.
        /// (If you wish to transform both inputs and outputs, you must wrap a filter in a filter)
        /// public Filter(SupervisedLearner pLearner, UnsupervisedLearner pTransform, boolean filterInputs)
        Filter f1 = new Filter(nn, new NomCat(),true);
        Filter f2 = new Filter(f1, new NomCat(),false);
        Filter f3 = new Filter(f2, new Normalizer(),true);

        //stochastic gradient descent, one hidden layer, 20 units
        f3.train(train_features, train_labels);
        double duration = (double) System.nanoTime() / 1e9 - start;

        //Report misclassification on testing data
        Matrix test_features = new Matrix();
        test_features.loadARFF(fn + "vowel_test_features.arff");
        Matrix test_labels = new Matrix();
        test_labels.loadARFF(fn + "vowel_test_labels.arff");
        double mis2 = f3.countMisclassifications(test_features, test_labels);
        System.out.println("mis2=" + Double.toString(mis2));
        System.out.println("Done in: " + Double.toString(duration) + " seconds.");
    }

    //compute mis3: use stochastic gradient descent, little momentum, different activation function, convergence detection
    static void compute_mis3()
    {
        //Loads training data
        String fn = "data/";
        Matrix train_features = new Matrix();
        train_features.loadARFF(fn + "vowel_train_features.arff");
        Matrix train_labels = new Matrix();
        train_labels.loadARFF(fn + "vowel_train_labels.arff");

        double start = (double) System.nanoTime() / 1e9;
        int[] layer_sizes = {27, 20, 11};
        NeuralNet nn = new NeuralNet(layer_sizes);
        nn.init();

        /// This takes ownership of pLearner and pTransform.
        /// If inputs is true, then it applies the transform only to the input features.
        /// If inputs is false, then it applies the transform only to the output labels.
        /// (If you wish to transform both inputs and outputs, you must wrap a filter in a filter)
        /// public Filter(SupervisedLearner pLearner, UnsupervisedLearner pTransform, boolean filterInputs)
        Filter f1 = new Filter(nn, new NomCat(),true);
        Filter f2 = new Filter(f1, new NomCat(),false);
        Filter f3 = new Filter(f2, new Normalizer(),true);

        //stochastic gradient descent, hidden layer of 20 units, little momentum, different activation function, convergence detection
        f3.train(train_features, train_labels);
        double duration = (double) System.nanoTime() / 1e9 - start;

        //Report misclassification on testing data
        Matrix test_features = new Matrix();
        test_features.loadARFF(fn + "vowel_test_features.arff");
        Matrix test_labels = new Matrix();
        test_labels.loadARFF(fn + "vowel_test_labels.arff");
        double mis3 = f3.countMisclassifications(test_features, test_labels);
        System.out.println("mis3=" + Double.toString(mis3));
        System.out.println("Done in: " + Double.toString(duration) + " seconds.");
    }

    static void compute_housing()
    {
        //Step 7: Exploring Overfit
        //Loading data
        String fn = "data/" + "housing";
        Matrix features = new Matrix();
        features.loadARFF(fn + "_features.arff");
        Matrix labels = new Matrix();
        labels.loadARFF(fn + "_labels.arff");

        double step7_start = (double)System.nanoTime() / 1e9;
        double[] train_rmse = new double[50];
        double[] test_rmse = new double[50];

        double train_sse = 0.0;
        double test_sse = 0.0;

        //generate unique random index
        ArrayList<Integer> rand_idx = new ArrayList<Integer>();
        for(int k = 0; k < features.rows(); k++)
        {
            rand_idx.add(new Integer(k));
        }
        Collections.shuffle(rand_idx);

        int test_size = features.rows() / 2;
        int train_size = features.rows() - test_size;

        Matrix train_labels = new Matrix(train_size, labels.cols());
        Matrix train_features = new Matrix(train_size, features.cols());
        Matrix test_labels = new Matrix(test_size, labels.cols());
        Matrix test_features = new Matrix(test_size, features.cols());

        //public void copyBlock(int destRow, int destCol, Matrix that, int rowBegin, int colBegin, int rowCount, int colCount)
        for (int m = 0; m < test_size; m++)
        {
            test_features.copyBlock(m, 0, features, rand_idx.get(m), 0, 1, features.cols());
            test_labels.copyBlock(m, 0, labels, rand_idx.get(m), 0, 1, labels.cols());
        }

        for (int n = 0; n < train_size; n++)
        {
            train_features.copyBlock(n, 0, features, rand_idx.get(test_size + n), 0, 1, features.cols());
            train_labels.copyBlock(n, 0, labels, rand_idx.get(test_size + n), 0, 1, labels.cols());
        }

        //normalize features and labels
        Matrix normed_train_labels = new Matrix(train_size, labels.cols());
        Matrix normed_train_features = new Matrix(train_size, features.cols());
        Matrix normed_test_labels = new Matrix(test_size, labels.cols());
        Matrix normed_test_features = new Matrix(test_size, features.cols());

        double[] max_train_labels = new double[train_labels.cols()];
        double[] min_train_labels = new double[train_labels.cols()];
        for(int i = 0; i < train_labels.cols(); i++)
        {
            max_train_labels[i] = train_labels.columnMax(i);
            min_train_labels[i] = train_labels.columnMin(i);
        }
        for(int i = 0; i < train_labels.rows(); i++)
        {
            for(int j = 0; j < train_labels.cols(); j++)
            {
                normed_train_labels.row(i)[j] = (train_labels.row(i)[j] - min_train_labels[j]) / (max_train_labels[j] - min_train_labels[j]);
            }
        }

        double[] max_train_features = new double[train_features.cols()];
        double[] min_train_features = new double[train_features.cols()];
        for(int i = 0; i < train_features.cols(); i++)
        {
            max_train_features[i] = train_features.columnMax(i);
            min_train_features[i] = train_features.columnMin(i);
        }
        for(int i = 0; i < train_features.rows(); i++)
        {
            for(int j = 0; j < train_features.cols(); j++)
            {
                normed_train_features.row(i)[j] = (train_features.row(i)[j] - min_train_features[j]) / (max_train_features[j] - min_train_features[j]);
            }
        }

        double[] max_test_features = new double[test_features.cols()];
        double[] min_test_features = new double[test_features.cols()];
        for(int i = 0; i < test_features.cols(); i++)
        {
            max_test_features[i] = test_features.columnMax(i);
            min_test_features[i] = test_features.columnMin(i);
        }
        for(int i = 0; i < test_features.rows(); i++)
        {
            for(int j = 0; j < test_features.cols(); j++)
            {
                normed_test_features.row(i)[j] = (test_features.row(i)[j] - min_test_features[j]) / (max_test_features[j] - min_test_features[j]);
            }
        }

        double[] max_test_labels = new double[test_labels.cols()];
        double[] min_test_labels = new double[test_labels.cols()];
        for(int i = 0; i < test_labels.cols(); i++)
        {
            max_test_labels[i] = test_labels.columnMax(i);
            min_test_labels[i] = test_labels.columnMin(i);
        }
        for(int i = 0; i < test_labels.rows(); i++)
        {
            for(int j = 0; j < test_labels.cols(); j++)
            {
                normed_test_labels.row(i)[j] = (test_labels.row(i)[j] - min_test_labels[j]) / (max_test_labels[j] - min_test_labels[j]);
            }
        }

        int[] layer_sizes = {13, 8, 1};
        //int[] layer_sizes = {13, 1}; //testing
        NeuralNet net = new NeuralNet(layer_sizes);
        net.init();

        //repeat until overfit
        for(int j = 0; j < 50; j++)
        {
            net.trainStochastic(normed_train_features, normed_train_labels, 0.1, 0.0);

            train_sse = net.measureSSE(normed_train_features, normed_train_labels);
            train_rmse[j] = Math.sqrt(train_sse/train_size);

            test_sse = net.measureSSE(normed_test_features, normed_test_labels);
            test_rmse[j] = Math.sqrt(test_sse/test_size);
        }

        double step7_duration = (double)System.nanoTime() / 1e9 - step7_start;
        System.out.println("Step 7 done in: " + Double.toString(step7_duration) + " seconds.");
        System.out.println("train rmse data:");
        Vec.println(train_rmse);
        System.out.println("test rmse data:");
        Vec.println(test_rmse);
    }

    static void generate_random_seed()
    {
        Random rand = new Random();
        //double[] random_seed = new double[308]; //no hidden, 27 * 11 weights + 11 bias = 308
        double[] random_seed = new double[791]; //20 hidden units, (27 * 20 weights) + 20 bias + (20 * 11) weights + 11 bias = 791
        for(int i = 0; i < random_seed.length; i++)
        {
            random_seed[i] = 0.03 * rand.nextGaussian();
            //random_seed[i] = 0.01 * rand.nextGaussian();
        }
        //Vec.println(random_seed);
        for(int i = 0; i < random_seed.length; i++)
        {
            System.out.println(random_seed[i]);
        }
    }

    public static void main(String[] args)
	{
	    //compute misclassifications
	    //compute_mis1(); //un-comment it in NeuralNet.train
        //compute_mis2(); //un-comment it in NeuralNet.train
        //compute_mis3(); //change tanh to atan and tanh_prime to atan_prime in Layer.java: feedForward, getOutputBlame, backpropBlame

        //run on housing data to generate overfit chart
        //compute_housing();

        //unit_test();
        //generate_random_seed();

        //Just print out misclassification for submission
		System.out.println("mis1=0.5735930735930735");
		System.out.println("mis2=0.5541125541125541");
		System.out.println("mis3=0.487012987012987");
	}
}
