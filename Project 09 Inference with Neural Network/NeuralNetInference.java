import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.lang.*;
import java.util.Collections;
import java.util.Random;

//Contains a collection of Layer objects that feed into each other.
public class NeuralNetInference extends SupervisedLearner
{
    ArrayList<Layer> m_layers = new ArrayList<Layer>();
    //double[] m_WeightsBiases;
    int[] layer_sizes;
    //V.setAll(0.0);
    //double[] m_blame;
    //double m_rand = Math.random();
    //Matrix V = new Matrix();

    Random rand = new Random();

    String name()
    {
        return "Neural Network";
    }

    NeuralNetInference(int[] layerSizes)
    {
        layer_sizes = layerSizes;
        int allWB = 0;
        for(int i = 1; i < layer_sizes.length; i++)
        {
            m_layers.add(new Layer(layer_sizes[i - 1], layer_sizes[i]));
            allWB += (layer_sizes[i] * layer_sizes[i - 1] + layer_sizes[i]);
            //i = 1, Layer(13, 8)
            //i = 2, Layer(8, 1)
        }
        //m_WeightsBiases = new double[allWB];
        //V.setSize(1000, 2);
        //V.setAll(0.0);
    }

    //add support for arbitrary activation function
    /*
    double atan (double input)
    {
        return Math.atan(input);
    }

    double tanh (double input)
    {
        return Math.tanh(input);
    }

    double atan_prime (double input)
    {
        return 1.0 / (input * input + 1.0);
    }

    double tanh_prime (double input)
    {
        return 1.0 - (tanh(input) * tanh(input));
    }
    */

    ///Set each weight and bias value to 0.03 * rand.normal();
    //reuse random seed
    void init()
    {
        //use loadARFF()
        //from Main.generate_random_seed()
//        Matrix random_seed = new Matrix();
        //random_seed.loadARFF("random_seed_7850.arff"); //no hidden layer
        //random_seed.loadARFF("random_seed_238510.arff"); //with hidden layer
        //random_seed.loadARFF("random_seed_545810.arff"); //with hidden layer
        //random_seed.loadARFF("random_seed_266610.arff"); //with hidden layer
        //random_seed.loadARFF("random_seed_795010.arff"); //with hidden layer
//        random_seed.loadARFF("random_seed_distinct.arff");
//        double[] weights_random_seed = new double[random_seed.rows()];
//        for(int i = 0; i < random_seed.rows(); i++)
//            weights_random_seed[i] = random_seed.row(i)[0];
//        setWeightsBiases(weights_random_seed);
        init_random();
    }

    void init_random()
    {

        //int idx = 0;
        for (int i = 0; i < layer_sizes.length - 1; i++)
        {
            Matrix w = new Matrix(layer_sizes[i + 1], layer_sizes[i]);
            double[] b = new double[layer_sizes[i + 1]];

            for (int x = 0; x < w.rows(); x++)
            {
                for (int y = 0; y < w.cols(); y++)
                {
                    //w.row(x)[y] = 0.03 * rand.nextGaussian();
                    w.row(x)[y] = Math.max(0.03, 1.0 / layer_sizes[i]) * rand.nextGaussian();
                    //w.row(x)[y] = 0.01 * rand.nextGaussian();
                    //w.row(x)[y] = 0.007 * x + 0.003 * y; //debug
                    //idx++;
                }
            }

            for (int z = 0; z < b.length; z++)
            {
                //b[z] = 0.03 * rand.nextGaussian();
                b[z] = Math.max(0.03, 1.0 / layer_sizes[i]) * rand.nextGaussian();
                //b[z] = 0.01 * rand.nextGaussian();
                //b[z] = 0.001 * z; //debug
                //idx++;
            }

            m_layers.get(i).setWeights(w);
            m_layers.get(i).setBiases(b);
        }
    }

    void setWeightsBiases(double[] wb)
    {
        if(wb == null)
            throw new RuntimeException("wb should not be null");

        int idx = 0;
        for (int i = 0; i < layer_sizes.length - 1; i++)
        {
            Matrix w = new Matrix(layer_sizes[i + 1], layer_sizes[i]);
            double[] b = new double[layer_sizes[i + 1]];

            for (int x = 0; x < w.rows(); x++)
            {
                for (int y = 0; y < w.cols(); y++)
                {
                    w.row(x)[y] = wb[idx];
                    idx++;
                }
            }

            for (int z = 0; z < b.length; z++)
            {
                b[z] = wb[idx];
                idx++;
            }

            m_layers.get(i).setWeights(w);
            m_layers.get(i).setBiases(b);
        }
    }

    //feed "in" to first layer and so on
    void predict(double[] in, double[] out)
    {
        if(in == null)
            throw new RuntimeException("in can't be null");
        m_layers.get(0).feedForward(in);
        for(int j = 1; j < m_layers.size(); j++)
        {
            m_layers.get(j).feedForward(m_layers.get(j-1).getActivation());
        }
        //System.out.println("out.length:");
        //System.out.println(out.length);
        Vec.copy(out, m_layers.get(m_layers.size() - 1).getActivation());
    }
    /*
	//indexing problem here
    void backPropagate()
    {
        //System.out.println("m_layers.size:");
        //System.out.println(m_layers.size());
        for(int i = m_layers.size() - 1; i > 1; i--)
        //for(int i = layer_sizes.length - 1; i > 0; i--)
            m_layers.get(i - 1).backPropagate(m_layers.get(i));
        //size 2, index: 0,1
        //
    }
    */

    //void backPropagate(double[] outBlame)
    void backPropagate()
    {
        //layer after output
        for(int i = m_layers.size() - 1; i > 0; i--)
            //m_layers.get(m_layers.size() - 2).updateBlame(m_layers.get(m_layers.size() - 1), outBlame);
            m_layers.get(i - 1).backpropBlame(m_layers.get(i));

        //next layer
        // if (m_layers.size() > 2) {
        //for (int i = m_layers.size() - 1; i > 1; i--)
        //  m_layers.get(i - 1).backpropBlame(m_layers.get(i));
        //}
    }

    void presentPattern(double[] features, double[] labels)
    //problem was here for more layers; fixed
    {
        // Feed the features forward through all the layers: L.feedForward
        m_layers.get(0).feedForward(features);
        //for next layers
        for(int j = 1; j < m_layers.size(); j++)
        {
            m_layers.get(j).feedForward(m_layers.get(j-1).getActivation());
        }

        // Compute blame for the net of the output layer: L.getBlame
        //double[] outBlame = m_layers.get(m_layers.size() - 1).getOutputBlame(labels);
        m_layers.get(m_layers.size() - 1).getOutputBlame(labels);

        // Backpropagate the blame across all the layers: L.backprop(Layer from)
        //m_layers.get(m_layers.size() - 1).backpropagate();
        if (m_layers.size() > 1)
            //backPropagate(outBlame);
            backPropagate();
        //m_layers.get(m_layers.size() - 1).backPropagate(m_layers.get(m_layers.size() - 1));

        // Call update_deltas for each layer
        // Problem was here for more layers; fixed
        m_layers.get(0).updateDeltas(features);

        //for next layers
        for (int j = 1; j < m_layers.size(); j++)
        {
            m_layers.get(j).updateDeltas(m_layers.get(j - 1).getActivation());
        }

    }

    // Train per epoch
    void trainEpoch(Matrix features, Matrix labels)
    {
        double learning_rate = 0.1;
        for(int j = 0; j < m_layers.size(); j++)
        //for(int j = 1; j < m_layers.size() - 1; j++)
        {
            m_layers.get(j).decayDeltas(0.0);
        }

        for(int i = 0; i < features.rows(); i++)
        {
            presentPattern(features.row(i), labels.row(i));
        }
        for(int j = 0; j < m_layers.size(); j++)
        //for(int j = 1; j < m_layers.size() - 1; j++)
        {
            m_layers.get(j).updateWeights(learning_rate);
        }
    }

    // Train with one batch (or mini-batch) of data
    //void trainBatch(Matrix features, Matrix labels, double learning_rate, double momentum)
    void trainBatch(Matrix features, Matrix labels, double learning_rate, double momentum)
    {

        //for(int j = 1; j < m_layers.size() - 1; j++)
        for(int j = 0; j < m_layers.size(); j++)
        {
            m_layers.get(j).decayDeltas(momentum);
        }


        for(int i = 0; i < features.rows(); i++)
        {
            presentPattern(features.row(i), labels.row(i));
        }

        //for(int j = 1; j < m_layers.size() - 1; j++)
        for(int j = 0; j < m_layers.size(); j++)
        {
            m_layers.get(j).updateWeights(learning_rate);
        }
    }

    // Perform one epoch of training with stochastic gradient descent
    void trainStochastic(Matrix features, Matrix labels, double learning_rate, double momentum)
    {
        int feat_rows = features.rows();
        int feat_cols = features.cols();
        int labels_rows = labels.rows();
        int labels_cols = labels.cols();

        //generate unique random row indexes
        ArrayList<Integer> rand_idx = new ArrayList<Integer>();
        for(int x = 0; x < feat_rows; x++)
        {
            rand_idx.add(new Integer(x));
        }
        Collections.shuffle(rand_idx);

        //copy features and labels in random order using Matrix.copyBlock
        //copyBlock(int destRow, int destCol, Matrix that, int rowBegin, int colBegin, int rowCount, int colCount)
        Matrix f = new Matrix(feat_rows, feat_cols);
        Matrix l = new Matrix(labels_rows, labels_cols);

        for (int m = 0; m < feat_rows; m++)
        {
            f.copyBlock(m, 0, features, rand_idx.get(m), 0, 1, feat_cols);
            l.copyBlock(m, 0, labels, rand_idx.get(m), 0, 1, labels_cols);
        }

        for(int i = 0; i < features.rows(); i++)
        {
            //for(int j = 1; j < m_layers.size() - 1; j++)
            for(int j = 0; j < m_layers.size(); j++)
            {
                m_layers.get(j).decayDeltas(momentum);
            }

            presentPattern(f.row(i), l.row(i));

            //L1 regularization
            /*
            //for(int j = 1; j < m_layers.size() - 1; j++)
            for(int j = 0; j < m_layers.size(); j++)
            {
                m_layers.get(j).regularize_L1(0.01, learning_rate);
            }
            */

            //L2 regularization
            /*
            //for(int j = 1; j < m_layers.size() - 1; j++)
            for(int j = 0; j < m_layers.size(); j++)
            {
                m_layers.get(j).regularize_L2(0.01, learning_rate);
            }
            */

            //for(int j = 1; j < m_layers.size() - 1; j++)
            for(int j = 0; j < m_layers.size(); j++)
            {
                m_layers.get(j).updateWeights(learning_rate);
            }
        }
    }

    void train(Matrix features, Matrix labels)
    {
        //trainStochastic(features, labels, 0.01, 0.0);
        //trainBatch(features, labels, 0.01, 0.0);
        //trainBatch(features, labels, 0.01);

        //use decaying learning rate
        double learning_rate = 0.01;
        //for(int i = 0; i < 100; i++)
        for(int i = 0; i < 5; i++)
        {
            trainStochastic(features, labels, learning_rate, 0.0);
            //trainBatch(features, labels, learning_rate, 0.0);
            learning_rate *= 0.98;
        }

        /*
        int train_size = (int) (0.7 * features.rows());
        int validation_size = features.rows() - train_size;

        //divide training data into two portions: 70 train, 30 validation
        Matrix features_train = new Matrix(train_size, features.cols());
        Matrix features_validation = new Matrix(validation_size, features.cols());
        Matrix labels_train = new Matrix(train_size, labels.cols());
        Matrix labels_validation = new Matrix(validation_size, labels.cols());

        //public void copyBlock(int destRow, int destCol, Matrix that, int rowBegin, int colBegin, int rowCount, int colCount)
        for (int m = 0; m < train_size; m++)
        {
            features_train.copyBlock(m, 0, features, m, 0, 1, features.cols());
            labels_train.copyBlock(m, 0, labels, m, 0, 1, labels.cols());
        }

        for (int n = 0; n < validation_size; n++)
        {
            features_validation.copyBlock(n, 0, features, train_size + n, 0, 1, features.cols());
            labels_validation.copyBlock(n, 0, labels, train_size + n, 0, 1, labels.cols());
        }

        //do until convergence is detected:
        double sse = 0.0;
        int burnIn = 50;
        int window = 10;
        double threshold = 0.05;

        for(int i = 1; i < burnIn; i++)
        {
            //choose one training method
            //trainStochastic(features_train, labels_train, 0.1, 0.0); //mis1 and mis2, no momentum
            trainStochastic(features_train, labels_train, 0.01, 0.1); //mis3, with momentum
            //trainBatch(features_train, labels_train, 0.03);
            //trainEpoch(features_train, labels_train);

            //this block is commented for compute_mis1 and compute_mis2 (no convergence detection)

            sse = this.measureSSE(features_validation, labels_validation);

            while(true)
            {
                double prevSSE = sse;

                for(int j = 1; j < window; j++)
                {
                    //choose one training method
                    //trainStochastic(features_train, labels_train, 0.01, 0.1); //mis2
                    trainStochastic(features_train, labels_train, 0.01, 0.1); //mis3
                    //trainBatch(features_train, labels_train, 0.03);
                    //trainEpoch(features_train, labels_train);
                }

                sse = this.measureSSE(features_validation, labels_validation);

                if((prevSSE - sse) / prevSSE < threshold || sse == 0.0)
                    break;
            }

        }
        */
    }


	void train_observation(Matrix X)
	{
	    Matrix V = new Matrix();
	    //V.setSize(1000, 2);
	    //System.out.println("Matrix X:");
	    //X.print();

		//64 x 48 pixels
	    int width = 64;
        int height = 48;
	    int channels = X.cols() / (width * height); //channels = 9216 / (64x48)
		//Initialize the MLP to have channels output units (not X.cols() outputs)
        //Random rand = new Random();
        int n = X.rows();
		//k = number of degrees of freedom in the system = 2
        V.setSize(n, 2);
        V.setAll(0.0);
		double learning_rate = 0.1;
        double[] input_blame = new double[4];
        //convergence detection?
		for(int j = 0; j < 10; j++)
		{
			for(int i = 0; i < 10000000; i++)
            //for(int i = 0; i < 100; i++) //debug
			{
				int t = rand.nextInt(X.rows());
				int p = rand.nextInt(width);
				int q = rand.nextInt(height);

                //debugging
                //int t = i % 1000;
                //int p = (i * 31) % 64;
                //int q = (i * 19) % 48;

				double[] features = {((double) p) / ((double) width), ((double) q) / ((double) height), V.row(t)[0], V.row(t)[1]};
				int s = channels * (width * q + p);
				//labels = the vector from X[t][s] to X[t][s + (channels - 1)];
                //labels = r,g,b
				double[] labels = {X.row(t)[s], X.row(t)[s+1], X.row(t)[s+2]};

                double[] pred = new double[labels.length];
                /*
                // Feed the features forward through all the layers: L.feedForward
                m_layers.get(0).feedForward(features);
                //for next layers
                for(int z = 1; z < m_layers.size(); z++)
                {
                    m_layers.get(z).feedForward(m_layers.get(z-1).getActivation());
                }
                */

                for(int z = 0; z < m_layers.size(); z++)
                {
                    m_layers.get(z).decayDeltas(0.0);
                }

				predict(features, pred);

				//compute the error on the output units
                m_layers.get(m_layers.size() - 1).getOutputBlame(labels);

                //do backpropagation to compute the errors of the hidden units
                backPropagate();

                //compute the blame terms for V[t]
                //blame on the input = transposed weights of the first layer * blame of the net of the first layer
                for(int x = 0; x < m_layers.get(0).m_weights.cols(); x++)
                {
                    double blame = 0.0;
                    for(int y = 0; y < m_layers.get(0).m_weights.rows(); y++)
                    {
                        blame += m_layers.get(0).m_weights.row(y)[x] * m_layers.get(0).m_blame[y];
                    }
                    input_blame[x] = blame;
                }

                //use gradient descent to refine the weights and bias values
                m_layers.get(0).updateDeltas(features);
                //for next layers
                for (int k = 1; k < m_layers.size(); k++)
                {
                    m_layers.get(k).updateDeltas(m_layers.get(k - 1).getActivation());
                }
                for(int m = 0; m < m_layers.size(); m++)
                {
                    m_layers.get(m).updateWeights(learning_rate);
                }
                //use gradient descent to update V[t]
                V.row(t)[0] += (learning_rate * input_blame[2]);
                V.row(t)[1] += (learning_rate * input_blame[3]);
                /*
                //debugging spew
                System.out.println("j=" + j + ", " + "i=" + i + ", ");
                for (int k = 0; k < m_layers.size() - 1; k++) {
                    System.out.println("Layer" + k + "_bias=");
                    Vec.println(m_layers.get(k).m_bias);
                    System.out.println("Layer" + k + "_weights=");
                    m_layers.get(k).m_weights.print();
                }
                System.out.println("t=" + t + ", " + "p=" + p + ", " + "q=" + q + ", ");
                System.out.println("features=");
                Vec.println(features);
                System.out.println("target=");
                Vec.println(labels);
                System.out.println("prediction=");
                Vec.println(pred);
                for (int k = 0; k < m_layers.size(); k++) {
                    System.out.println("Layer" + k + "_blame=");
                    Vec.println(m_layers.get(k).m_blame);
                }
                System.out.println("input_blame=");
                Vec.println(input_blame);
                System.out.println("Updated V[t]=");
                Vec.println(V.row(t));
                */
			}
			learning_rate *= 0.75;
		}
		//save V to arff file
        V.saveARFF("Matrix_V.arff");
	}

    void generate_image(double[] state, String filename) throws IOException
    {
        int width = 64; int height = 48;
        BufferedImage new_image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);

        File outputFile = new File(filename);

        double[] in = new double[4];
        in[2] = state[0];
        in[3] = state[1];
        double[] out = new double[3];

        for(int y = 0; y < height; y++)
        {
            in[1] = ((double) y) / ((double) height);
            for(int x = 0; x < width; x++)
            {
                in[0] = ((double)x) / ((double) width);
                this.predict(in, out);
                //Vec.print(out);
                int color = rgbToInt((int)(Math.round(out[0] * ((double) 256))), (int) (Math.round(out[1] * ((double) 256))), (int) (Math.round(out[2] * ((double) 256))));
                //int color = rgbToInt((int)(out[0] * (double) 256), (int) (out[1] * (double) 256), (int) (out[2] * (double) 256));
                new_image.setRGB(x, y, color);
            }
        }
        ImageIO.write(new_image, "png", outputFile);
    }

    int rgbToInt (int red, int green, int blue)
    {

        int rgb = red;
        rgb = (rgb << 8) + green;
        rgb = (rgb << 8) + blue;
        return rgb;

        //return 0xff000000 | ((red & 0xff) << 16)) | ((green & 0xff) << 8) | ((blue & 0xff));
    }
}