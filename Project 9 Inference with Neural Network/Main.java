import java.lang.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;


class Main
{   /*
    static int rgbToInt (int red, int green, int blue)
    {

        int rgb = red;
        rgb = (rgb << 8) + green;
        rgb = (rgb << 8) + blue;
        return rgb;

        //return 0xff000000 | ((red & 0xff) << 16)) | ((green & 0xff) << 8) | ((blue & 0xff));
    }

    static void generate_image(double[] state, String filename, NeuralNetInference nn) throws IOException
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
                nn.predict(in, out);
                //Vec.print(out);
                int color = rgbToInt((int)(Math.round(out[0] * (double) 256)), (int) (Math.round(out[1] * (double) 256)), (int) (Math.round(out[2] * (double) 256)));
                new_image.setRGB(x, y, color);
            }
        }
        ImageIO.write(new_image, "png", outputFile);
    }

    static void generate_image_test() throws IOException
    {
        int width = 64; int height = 48;
        BufferedImage new_image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);

        File outputFile = new File("frame_test.png");
        //Loading data
        String fn = "data/";
        Matrix observations = new Matrix();
        observations.loadARFF(fn + "observations.arff");
        double[] rgb_values = observations.row(222);
        int idx = 0;
        int[] color = new int[width * height];

        //for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width * 3; x+=3)
            //for(int y = 0; y < height  * 3; y+=3)
            {
                //color[idx] = rgbToInt((int) rgb_values[y], (int) rgb_values[y+1], (int) rgb_values[y+2]);
                color[idx] = rgbToInt((int) rgb_values[x], (int) rgb_values[x+1], (int) rgb_values[x+2]);
                idx++;
            }
        }

        int pos = 0;
        for(int y = 0; y < height; y++)
        //for(int x = 0; x < width; x++)
        {
            for(int x = 0; x < width; x++)
            //for(int y = 0; y < height; y++)
            {
                new_image.setRGB(x, y, color[pos]);
                pos++;
            }
        }

        ImageIO.write(new_image, "png", outputFile);
    }
    */

    static void do_observation()
    {
        //Loading data
        String fn = "data/";
        Matrix observations = new Matrix();
        observations.loadARFF(fn + "observations.arff");
        //Matrix actions = new Matrix();
        //actions.loadARFF(fn + "actions.arff");

        //normalization: divide all observation values by 256
        //don't forget to multiply by 256 before generating an image
        for (int i = 0; i < observations.rows(); i++)
        {
            for (int j = 0; j < observations.cols(); j++)
            {
                observations.row(i)[j] /= (double) 256;
            }
        }

        //Observation function
        double start1 = (double)System.nanoTime() / 1e9;
        int[] layer_sizes1 = {4, 12, 12, 3};
        //int[] layer_sizes1 = {4, 48, 48, 3};
        NeuralNetInference nn1 = new NeuralNetInference(layer_sizes1);
        nn1.init();
        nn1.train_observation(observations);
        //it saves Matrix V to an arff file

        double duration1 = (double)System.nanoTime() / 1e9 - start1;
        System.out.println("Observation function done in: " + Double.toString(duration1) + " seconds.");
    }

    static void do_transition() throws IOException
    {
        //Loading data
        String fn = "data/";
        Matrix observations = new Matrix();
        observations.loadARFF(fn + "observations.arff");
        //Matrix actions = new Matrix();
        //actions.loadARFF(fn + "actions.arff");

        //normalization: divide all observation values by 256
        //don't forget to multiply by 256 before generating an image
        for (int i = 0; i < observations.rows(); i++)
        {
            for (int j = 0; j < observations.cols(); j++)
            {
                observations.row(i)[j] /= (double) 256;
            }
        }

        //Observation function
        double start1 = (double)System.nanoTime() / 1e9;
        int[] layer_sizes1 = {4, 12, 12, 3};
        //int[] layer_sizes1 = {4, 128, 128, 3};
        NeuralNetInference nn1 = new NeuralNetInference(layer_sizes1);
        nn1.init();
        nn1.train_observation(observations); //it saves Matrix V to an arff file

        double duration1 = (double)System.nanoTime() / 1e9 - start1;
        System.out.println("Observation function done in: " + Double.toString(duration1) + " seconds.");

        Matrix V = new Matrix();
        V.loadARFF("Matrix_V.arff");

        //Loading data
        Matrix A = new Matrix();
        A.loadARFF(fn + "actions.arff");

        //Transition function
        double start2 = (double)System.nanoTime() / 1e9;
        int[] layer_sizes2 = {6, 6, 2};
        //int[] layer_sizes2 = {6, 24, 2};
        NeuralNet nn2 = new NeuralNet(layer_sizes2);
        nn2.init();
        Filter f1 = new Filter(nn2, new NomCat(), true);
        Filter f2 = new Filter(f1, new Normalizer(), true);
        Filter f3 = new Filter(f2, new Normalizer(), false);

        Matrix next_V = new Matrix();
        next_V.setSize(V.rows() - 1, V.cols());
        next_V.copyBlock(0, 0, V, 1, 0, V.rows() - 1, V.cols());

        //The features of this data consist of each of the rows in V (except the last one),
        // and the action that was performed in that state.
        Matrix features = new Matrix();
        features.setSize(V.rows() - 1, V.cols() + A.cols());
        features.copyBlock(0, 0, V, 0, 0, V.rows() - 1, V.cols());
        features.copyBlock(0, V.cols(), A, 0, 0, V.rows() - 1, A.cols());
        //The labels consist of the next row in V, because you are trying to predict the state that will follow.
        // (I found that I obtain better results when I predict the difference between the next state and the current state,
        // and then add this difference to the current state to predict the next one.)
        Matrix labels = new Matrix();
        labels.setSize(next_V.rows(), next_V.cols());
        //labels.copyBlock(0, 0, next_V, 0, 0, next_V.rows(), next_V.cols()); //predict the next state

        //predict the difference
        for (int i = 0; i < labels.rows(); i++)
        {
            for (int j = 0; j < labels.cols(); j++)
            {
                labels.row(i)[j] = next_V.row(i)[j] - V.row(i)[j];
            }
        }

        f3.train(features, labels);
        //f1.train(features, labels);

        //Generate images
        //Initialize the state vector to the first row in V.
        //(This is the state where the crane is in the central position.)
        //Feed this state through the observation function to generate a predicted image
        //and save it as "frame0.svg" or "frame0.png". This image should depict the crane in its central starting position.
        double[] state = new double[2];
        Vec.copy(state, V.row(0));
        //System.out.println("state of frame 0: ");
        //Vec.print(state);
        nn1.generate_image(state, "frame0.png");

        //Use your model to predict how state changes as you perform action 'a' five times.
        //After each action, save the predicted images as "frame1.svg", "frame2.svg", etc.
        Matrix feature = new Matrix();
        feature.setSize(11, 6);
        double[] a = {1.0, 0.0, 0.0, 0.0};
        Vec.copy(feature.row(0), Vec.concatenate(state, a));
        double[] state_diff = new double[2];

        for (int i = 0; i < 5; i++)
        {
            nn2.predict(feature.row(i), state_diff);
            Vec.add(state, state_diff);
            //f3.predict(feature.row(i), state);
            //System.out.println("state of frame" + (i + 1) + ": ");
            //Vec.print(state);
            nn1.generate_image(state, "frame" + (i + 1) + ".png");
            Vec.copy(feature.row(i + 1), Vec.concatenate(state, a));
        }

        //Next, perform action 'c' five times. After each action, save the images as "frame6.svg", "frame7.svg", etc.
        //(In other words, simulate moving the crane left five times, and then up five times.)
        double[] c = {0.0, 0.0, 1.0, 0.0};
        for (int i = 5; i < 10; i++)
        {
            nn2.predict(feature.row(i), state_diff);
            Vec.add(state, state_diff);
            //f3.predict(feature.row(i), state);
            //System.out.println("state of frame" + (i + 1) + ": ");
            //Vec.print(state);
            nn1.generate_image(state, "frame" + (i + 1) + ".png");
            Vec.copy(feature.row(i + 1), Vec.concatenate(state, c));
        }

        double duration2 = (double)System.nanoTime() / 1e9 - start2;
        System.out.println("Transition function done in: " + Double.toString(duration2) + " seconds.");

    }

	public static void main(String[] args) throws IOException
	{	
        //do the unsupervised, saves Matrix V to an arff file, plot crane wanders in its state space
		//do_observation();

        //do supervise, predict action, generate crane movement images
        do_transition();

        //generate_image_test();
	}
}
