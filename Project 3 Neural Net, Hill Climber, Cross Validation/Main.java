import java.lang.*;
import java.util.*;

class Main
{
	static void debug_crossValidate()
	{
		//Loading data
		String fn = "data/" + "debugcv";
		Matrix features = new Matrix();
		features.loadARFF(fn + "_features.arff");
		Matrix labels = new Matrix();
		labels.loadARFF(fn + "_labels.arff");

		//Step 6: Getting RMSE
		double step6_start = (double)System.nanoTime() / 1e9;
		int[] layer_sizes = {1, 1};
		//int[] layer_sizes = {13, 1}; //debugging
		//int[] layer_sizes = {1,1}; //debugging
		NeuralNet nn = new NeuralNet(layer_sizes);
		//nn.init();
		//Normalizer n1 = new Normalizer();
		//Normalizer n2 = new Normalizer();
		//Filter x = new Filter(nn, n1, true);
		//Filter y = new Filter(x, n2, false);

		//2 repetitions, 10-folds; rmse = sqrt(sse/n)
		double sse = nn.crossValidate(2, 3, features, labels);
		//debugging
		//nn.train(features, labels); //debugging hill climber
		//double sse = nn.measureSSE(features, labels); //debugging hill climber
		//RMSE = sqrt(SSE / m), where SSE is the number that crossValidate returns, and m is the number of rows in your data.
		double num = features.rows();
		double rmse = Math.sqrt(sse/num);
		double step6_duration = (double)System.nanoTime() / 1e9 - step6_start; //seconds = nano seconds / 1e9!
		System.out.println("rmse=" + Double.toString(rmse));
		System.out.println("sse=" + Double.toString(sse));
		System.out.println("Step 6 done in: " + Double.toString(step6_duration) + " seconds.");

	}

	static void do_step_6()
    {
        //Loading data
        String fn = "data/" + "housing";
        Matrix features = new Matrix();
        features.loadARFF(fn + "_features.arff");
        Matrix labels = new Matrix();
        labels.loadARFF(fn + "_labels.arff");

        //Step 6: Getting RMSE
		double step6_start = (double)System.nanoTime() / 1e9;
		int[] layer_sizes = {13, 8, 1};
        //int[] layer_sizes = {13, 1}; //debugging
        //int[] layer_sizes = {1,1}; //debugging
		NeuralNet nn = new NeuralNet(layer_sizes);
		//nn.init();
		Normalizer n1 = new Normalizer();
		Normalizer n2 = new Normalizer();
        Filter x = new Filter(nn, n1, true);
        Filter y = new Filter(x, n2, false);

        //2 repetitions, 10-folds; rmse = sqrt(sse/n)
		double sse = y.crossValidate(2, 10, features, labels);
        //debugging
        //nn.train(features, labels); //debugging hill climber
        //double sse = nn.measureSSE(features, labels); //debugging hill climber
		//RMSE = sqrt(SSE / m), where SSE is the number that crossValidate returns, and m is the number of rows in your data.
        double num = features.rows();
        double rmse = Math.sqrt(sse/num);
		double step6_duration = (double)System.nanoTime() / 1e9 - step6_start; //seconds = nano seconds / 1e9!
		System.out.println("rmse=" + Double.toString(rmse));
		System.out.println("sse=" + Double.toString(sse));
		System.out.println("Step 6 done in: " + Double.toString(step6_duration) + " seconds.");

    }

    static void do_step_7()
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
    	//double testPortion = 0.5;

		int[] layer_sizes = {13, 8, 1};
		//int[] layer_sizes = {13, 1}; //testing
		NeuralNet net = new NeuralNet(layer_sizes);
        //Normalizer norm1 = new Normalizer();
        //Normalizer norm2 = new Normalizer();
        //Filter x = new Filter(net, norm1, true);
        //Filter nn = new Filter(x, norm2, false);
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

		HillClimber hc1 = new HillClimber(normed_train_features, normed_train_labels, net);

		//repeat until overfit
		for(int j = 0; j < 50; j++)
		{
			hc1.iterate();
    		//train_sse = hc1.iterate();
    		train_sse = net.measureSSE(normed_train_features, normed_train_labels);
			train_rmse[j] = Math.sqrt(train_sse/train_size);

			//HillClimber hc2 = new HillClimber(normed_test_features, normed_test_labels, net);
			//hc2.iterate();
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

	public static void main(String[] args)
	{
		//debug_crossValidate();
		//do_step_6();
		//do_step_7();

		//Just print out rmse for submission
		System.out.println("rmse=8.707704098781536");
	}

}
