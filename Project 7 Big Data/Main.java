import java.lang.*;
import java.util.*;

class Main
{
    static void do_project_07()
    {
        //Loads training data
        String fn = "data/";
        Matrix train_features = new Matrix();
        train_features.loadARFF(fn + "train_feat.arff");
        Matrix train_labels = new Matrix();
        train_labels.loadARFF(fn + "train_lab.arff");
        Matrix test_features = new Matrix();
        test_features.loadARFF(fn + "test_feat.arff");
        Matrix test_labels = new Matrix();
        test_labels.loadARFF(fn + "test_lab.arff");

        double start = (double) System.nanoTime() / 1e9;
        //int[] layer_sizes = {784, 1000, 10};
        //int[] layer_sizes = {784, 200, 80, 30, 10};
        //int[] layer_sizes = {784, 300, 100, 10};
        //int[] layer_sizes = {784, 500, 150, 10};
        //int[] layer_sizes = {784, 1000, 500, 10};
        //int[] layer_sizes = {784, 800, 500, 300, 10};
        //int[] layer_sizes = {784, 500, 500, 2000, 30, 10};
        //int[] layer_sizes = {784, 300, 10};
        //int[] layer_sizes = {784, 80, 10};
        //int[] layer_sizes = {784, 800, 10};
        //int[] layer_sizes = {784, 800, 800, 10};
        //int[] layer_sizes = {784, 1200, 600, 10};
        //int[] layer_sizes = {784, 1500, 1500, 10};
        //int[] layer_sizes = {784, 1000, 1000, 10};
        //int[] layer_sizes = {784, 10}; //testing
        Random random = new Random();
        int[] layer_sizes = {784, 80, 30, 10};
        NeuralNet nn = new NeuralNet(layer_sizes, random);
        //nn.init();

        //Filter f1 = new Filter(nn, new Normalizer(), true);
        //Filter f2 = new Filter(f1, new NomCat(), false);
        Filter f1 = new Filter(nn, new NomCat(), false);
        Filter f2 = new Filter(f1, new Normalizer(), true);
        int mis = 0;
        int tr_mis = 0;

        for(int i = 1; i < 10; i++)
        {
            double start1 = (double) System.nanoTime() / 1e9;
            f2.train(train_features, train_labels);
            mis = f2.countMisclassifications(test_features, test_labels);
            double misrate = (double) mis / test_features.rows();
            tr_mis = f2.countMisclassifications(train_features, train_labels);
            double tr_misrate = (double) tr_mis / train_features.rows();
            double duration1 = (double) System.nanoTime() / 1e9 - start1;
            System.out.println("test mis = " + Integer.toString(mis) + " (" + Double.toString(misrate) + ")");
            System.out.println("train mis = " + Integer.toString(tr_mis) + " (" + Double.toString(tr_misrate) + ")");
            System.out.println("Epoch " + i + " done in: " + Double.toString(duration1) + " seconds.");
            /*
            while(true)
            {
                i++;
                double start2 = (double) System.nanoTime() / 1e9;
                f2.train(train_features, train_labels);
                mis = f2.countMisclassifications(test_features, test_labels);
                misrate = (double) mis / test_features.rows();
                tr_mis = f2.countMisclassifications(train_features, train_labels);
                tr_misrate = (double) tr_mis / train_features.rows();
                double duration2 = (double) System.nanoTime() / 1e9 - start2;
                System.out.println("test mis = " + Integer.toString(mis) + " (" + Double.toString(misrate) + ")");
                System.out.println("train mis = " + Integer.toString(tr_mis) + " (" + Double.toString(tr_misrate) + ")");
                System.out.println("Epoch " + i + " done in: " + Double.toString(duration2) + " seconds.");
                if(mis < 400)
                    break;
            }
            */
        }

        double duration = (double) System.nanoTime() / 1e9 - start;

        //Report misclassification on testing data
        System.out.println("mis=" + Integer.toString(mis));
        System.out.println("Project 7 done in: " + Double.toString(duration) + " seconds.");
    }

    //generate distinct random seed per layer
    //project 7: "max(0.03, 1.0 / fan_in) * rand.normal()", where fan_in is the number of weights feeding into the layer
    static void generate_random_seed_7()
    {
        int[] layer_sizes = {784, 80, 30, 10};
        ArrayList<Layer> m_layers = new ArrayList<Layer>();
        Random rand = new Random();
        int allWB = 0;
        for(int i = 1; i < layer_sizes.length; i++)
        {
            m_layers.add(new Layer(layer_sizes[i - 1], layer_sizes[i]));
            allWB += (layer_sizes[i] * layer_sizes[i - 1] + layer_sizes[i]);
            //i = 1, Layer(13, 8)
            //i = 2, Layer(8, 1)
        }
        double[] random_seed = new double[allWB];
        int[] layerWB = new int[m_layers.size()];
        int idx = 0;
        //first layer
        for(int j = 0; j < ((layer_sizes[0] * layer_sizes[1]) + layer_sizes[1]); j++)
        {
            double mag = Math.max(0.05, 1.0 / layer_sizes[0]);
            random_seed[idx] = mag * rand.nextGaussian();
            idx++;
        }
        for(int i = 0; i < m_layers.size(); i++)
        {
            layerWB[i] = m_layers.get(i).m_weights.cols() * m_layers.get(i).m_weights.rows() + m_layers.get(i).m_bias.length;
        }
        //next layers
        for(int i = 1; i < m_layers.size(); i++)
        {
            for(int j = 0; j < layerWB[i]; j++)
            {
                double mag = Math.max(0.05, 1.0 / (m_layers.get(i-1).m_weights.cols() * m_layers.get(i-1).m_weights.rows() + m_layers.get(i-1).m_bias.length));
                random_seed[idx] = mag * rand.nextGaussian();
                idx++;
            }
        }
        //printout arff header
        System.out.println("@RELATION random seed");
        System.out.println("@ATTRIBUTE x real");
        System.out.println("@DATA");
        for(int i = 0; i < random_seed.length; i++)
        {
            System.out.println(random_seed[i]);
        }
    }

    //project 4: mag * rand.normal(), double mag = max(0.03, 1.0 / layer.inputCount());
    static void generate_random_seed_4()
    {
        //int[] layer_sizes = {784, 300, 80, 30, 10};
        int[] layer_sizes = {784, 80, 30, 10};
        //int[] layer_sizes = {784, 200, 80, 30, 10};
        ArrayList<Layer> m_layers = new ArrayList<Layer>();
        Random rand = new Random();
        int allWB = 0;
        for(int i = 1; i < layer_sizes.length; i++)
        {
            m_layers.add(new Layer(layer_sizes[i - 1], layer_sizes[i]));
            allWB += (layer_sizes[i] * layer_sizes[i - 1] + layer_sizes[i]);
            //i = 1, Layer(13, 8)
            //i = 2, Layer(8, 1)
        }
        double[] random_seed = new double[allWB];
        int[] layerWB = new int[m_layers.size()];
        int idx = 0;
        for(int i = 0; i < m_layers.size(); i++)
        {
            layerWB[i] = m_layers.get(i).m_weights.cols() * m_layers.get(i).m_weights.rows() + m_layers.get(i).m_bias.length;
            for(int j = 0; j < layerWB[i]; j++)
            {
                double mag = Math.max(0.05, 1.0 / m_layers.get(i).m_weights.cols());
                random_seed[idx] = mag * rand.nextGaussian();
                idx++;
            }
        }
        //printout arff header
        System.out.println("@RELATION random seed");
        System.out.println("@ATTRIBUTE x real");
        System.out.println("@DATA");
        for(int i = 0; i < random_seed.length; i++)
        {
            System.out.println(random_seed[i]);
        }
    }

    static void do_unit_test()
    {
        int[] layer_sizes = {2, 2, 2, 2, 2};
        Random random = new Random();
        NeuralNet nn = new NeuralNet(layer_sizes, random);
        nn.m_layers.get(0).m_weights.setAll(0.1);
        nn.m_layers.get(0).m_bias = new double[]{0.0, 0.1};
        nn.m_layers.get(1).m_weights.setAll(0.1);
        nn.m_layers.get(1).m_bias = new double[]{0.0, 0.1};
        nn.m_layers.get(2).m_weights.setAll(0.1);
        nn.m_layers.get(2).m_bias = new double[]{0.0, 0.1};
        nn.m_layers.get(3).m_weights.setAll(0.1);
        nn.m_layers.get(3).m_bias = new double[]{0.0, 0.1};
        Matrix features = new Matrix(1,2);
        Matrix labels = new Matrix(1,2);
        features.setAll(0.1);
        labels.setAll(0.1);
        nn.trainStochastic(features, labels, 0.1, 0.0);
        for (int i = 0; i < nn.m_layers.size() ; i++)
        {
            System.out.println("Layer" + i + ":");
            System.out.println("m_weights:");
            nn.m_layers.get(i).m_weights.print();
            System.out.println("m_bias:");
            Vec.println(nn.m_layers.get(i).m_bias);
            System.out.println("m_net:");
            Vec.println(nn.m_layers.get(i).m_net);
            System.out.println("m_activation:");
            Vec.println(nn.m_layers.get(i).m_activation);
            System.out.println("m_blame:");
            Vec.println(nn.m_layers.get(i).m_blame);
            System.out.println("m_weightsDelta:");
            nn.m_layers.get(i).m_weightsDelta.print();
            System.out.println("m_biasDelta:");
            Vec.println(nn.m_layers.get(i).m_biasDelta);
        }
        nn.trainStochastic(features, labels, 0.1, 0.0);
        for (int i = 0; i < nn.m_layers.size() ; i++)
        {
            System.out.println("Layer" + i + ":");
            System.out.println("m_weights:");
            nn.m_layers.get(i).m_weights.print();
            System.out.println("m_bias:");
            Vec.println(nn.m_layers.get(i).m_bias);
            System.out.println("m_net:");
            Vec.println(nn.m_layers.get(i).m_net);
            System.out.println("m_activation:");
            Vec.println(nn.m_layers.get(i).m_activation);
            System.out.println("m_blame:");
            Vec.println(nn.m_layers.get(i).m_blame);
            System.out.println("m_weightsDelta:");
            nn.m_layers.get(i).m_weightsDelta.print();
            System.out.println("m_biasDelta:");
            Vec.println(nn.m_layers.get(i).m_biasDelta);
        }
    }

    static void do_project_07_debug()
    {
        //Loads training data
        String fn = "data/";
        Matrix train_features = new Matrix();
        train_features.loadARFF(fn + "train_feat.arff");
        Matrix train_labels = new Matrix();
        train_labels.loadARFF(fn + "train_lab.arff");
        Matrix test_features = new Matrix();
        test_features.loadARFF(fn + "test_feat.arff");
        Matrix test_labels = new Matrix();
        test_labels.loadARFF(fn + "test_lab.arff");
        Random random = new Random();
        int[] layer_sizes = {784, 80, 30, 10};
        NeuralNet nn = new NeuralNet(layer_sizes, random);
        //nn.init();

        //Filter f1 = new Filter(nn, new Normalizer(), true);
        //Filter f2 = new Filter(f1, new NomCat(), false);
        Filter f1 = new Filter(nn, new NomCat(), false);
        Filter f2 = new Filter(f1, new Normalizer(), true);

        for (int j = 0; j < 5; j++)
        {
            System.out.println("j=" + j);
            f2.train(train_features, train_labels);
            for (int i = 0; i < nn.m_layers.size(); i++) {
                System.out.println("Layer" + i + ":");
                System.out.println("m_weights:");
                nn.m_layers.get(i).m_weights.print();
                System.out.println("m_bias:");
                Vec.println(nn.m_layers.get(i).m_bias);
                System.out.println("m_net:");
                Vec.println(nn.m_layers.get(i).m_net);
                System.out.println("m_activation:");
                Vec.println(nn.m_layers.get(i).m_activation);
                System.out.println("m_blame:");
                Vec.println(nn.m_layers.get(i).m_blame);
                System.out.println("m_weightsDelta:");
                nn.m_layers.get(i).m_weightsDelta.print();
                System.out.println("m_biasDelta:");
                Vec.println(nn.m_layers.get(i).m_biasDelta);
            }
        }

    }

    public static void main(String[] args)
    {
        do_project_07();
        //do_project_07_debug();
        //generate_random_seed_4();
        //generate_random_seed_7();
        //do_unit_test();

        //do not include data in submission
        //Just print out misclassification for submission
        //System.out.println("mis=386");
    }
}
