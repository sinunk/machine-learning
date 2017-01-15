import java.lang.*;
import java.util.*;

class Main
{
    static void do_project_11()
    {
        //Loading data
        String fn = "data/";
        Matrix features = new Matrix();
        features.loadARFF(fn + "features.arff");
        Matrix labels = new Matrix();
        labels.loadARFF(fn + "labels.arff");
        Matrix test_features = new Matrix();
        test_features.loadARFF(fn + "test_features.arff");

        //Train
        double start = (double)System.nanoTime() / 1e9;
        int[] layer_sizes = {1, 100, 1};
        NeuralNet nn = new NeuralNet(layer_sizes);
        nn.init();
        Filter f1 = new Filter(nn, new Normalizer(), true);
        Filter f2 = new Filter(f1, new Normalizer(), false);
        for (int i = 0; i < 15 ; i++) {
            f2.train(features, labels);
        }
        //saves prediction without regularization, done
        //saves prediction with L1 regularization, un-comment in NeuralNet.trainStochastic
        //saves prediction with L2 regularization
        double[] prediction = new double[356];
        for(int i = 0; i < 256; i++)
        {
            double[] pred = new double[1];
            f2.predict(features.row(i), pred);
            prediction[i] = pred[0];
        }

        for(int i = 0; i < 100; i++)
        {
            double[] pred = new double[1];
            f2.predict(test_features.row(i), pred);
            prediction[i + 256] = pred[0];
        }

        for(int i = 0; i < prediction.length; i++)
        {
            System.out.println(prediction[i]);
        }
        double duration = (double) System.nanoTime() / 1e9 - start;
        //System.out.println("Project 11 done in: " + Double.toString(duration) + " seconds.");

    }

    static void generate_weights()
    {
        double[] weights_seed = new double[301];
        for (int i = 0; i < 50; i++)
        {
            weights_seed[i] = (i + 1) * 2 * Math.PI;
            weights_seed[i + 50] = Math.PI;
        }

        for (int i = 0; i < 50; i++)
        {
            weights_seed[i + 100] = (i + 1) * 2 * Math.PI;
            weights_seed[i + 150] = Math.PI / 2;
        }
        Random rand = new Random();
        for(int i = 200; i < 301; i++)
        {
            weights_seed[i] = 0.03 * rand.nextGaussian();
            //random_seed[i] = 0.01 * rand.nextGaussian();
        }
        for(int i = 0; i < weights_seed.length; i++)
        {
            System.out.println(weights_seed[i]);
        }
    }

    static void generate_features()
    {
        double[] features = new double[256];
        for(int i = 0; i < 256; i++)
            features[i] = (double) i/256;

        for(int i = 0; i < 256; i++)
            System.out.println(features[i]);
    }

    static void generate_test_features()
    {
        double[] features = new double[100];
        for(int i = 0; i < 100; i++)
            features[i] = (double) (i+256)/256;

        for(int i = 0; i < 100; i++)
            System.out.println(features[i]);
    }

    public static void main(String[] args)
	{
        do_project_11();
        //generate_weights();
        //generate_features();
        //generate_test_features();
	}
}
