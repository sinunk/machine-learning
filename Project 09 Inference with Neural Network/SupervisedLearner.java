// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.ArrayList;
import java.util.Collections;

abstract class SupervisedLearner 
{
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels);

	/// Make a prediction
	abstract void predict(double[] in, double[] out);
	
	/// For making chart in Project 3 Step 7
	//abstract void trainEpoch(Matrix features, Matrix labels);
	
	/// Measures the misclassifications with the provided test data
	int countMisclassifications(Matrix features, Matrix labels)
	{
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		double[] pred = new double[labels.cols()];
		int mis = 0;
		for(int i = 0; i < features.rows(); i++)
		{
			double[] feat = features.row(i);
			predict(feat, pred);
			double[] lab = labels.row(i);
			for(int j = 0; j < lab.length; j++)
			{
				if(pred[j] != lab[j])
					mis++;
			}
		}
		return mis;
	}
	
	/// Measures Sum Squared Error between predicted and label values
	double measureSSE (Matrix features, Matrix labels)
	{
	if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
	
	double[] pred = new double[labels.cols()];
	double sse = 0.0;
	
		for(int i = 0; i < features.rows(); i++)
		{
			double[] feat = features.row(i);
			predict(feat, pred);
			double[] lab = labels.row(i);
				
			for(int j = 0; j < lab.length; j++)
			{
				double d = pred[j] - lab[j];
				sse += (d * d);
			}
			//debugging
			/*
			System.out.println("feat=");
			Vec.println(feat);
			System.out.println(", lab=");
			Vec.println(lab);
			System.out.println(", pred=");
			Vec.println(pred);
			System.out.println(", sse=");
			System.out.println(sse);
			*/
		}
	return sse;
	}
	
	/// Do cross validation with specified folds and reps
	double crossValidate(int reps, int folds, Matrix features, Matrix labels)
	{
		int feat_rows = features.rows();
		int feat_cols = features.cols();
		int labels_rows = labels.rows();
		int labels_cols = labels.cols();
  		int test_row = feat_rows / folds;
		int train_row = feat_rows - test_row;
		
		if(feat_rows != labels_rows)
			throw new IllegalArgumentException("Mismatching number of rows");
		
		if(folds < 2)
			throw new IllegalArgumentException("Folds must be at least 2");
		
		if(folds > feat_rows)
			throw new IllegalArgumentException("Number of folds can't exceed number of datapoints");
		
		double sse = 0.0;
	
		for(int i = 0; i < reps; i++)
		{
			//generate unique random index
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
		
			Matrix train_f = new Matrix(train_row, feat_cols);
			Matrix train_l = new Matrix(train_row, labels_cols);
			Matrix fi = new Matrix(test_row, feat_cols);
			Matrix li = new Matrix(test_row, labels_cols);
	
			for (int k = 0; k < folds; k++)
			{
				fi.copyBlock(0, 0, f, 0 + test_row * k, 0, test_row, feat_cols);
				li.copyBlock(0, 0, l, 0 + test_row * k, 0, test_row, labels_cols);
			
				if (k == 0)
				{
					train_f.copyBlock(0, 0, f, test_row, 0, train_row, feat_cols);
					train_l.copyBlock(0, 0, l, test_row, 0, train_row, labels_cols);
				}

				if (k > 0)
				{
					train_f.copyBlock(0, 0, f, 0, 0, test_row * k, feat_cols);
					train_l.copyBlock(0, 0, l, 0, 0, test_row * k , labels_cols);
				}
				if (test_row * k < train_row)
				{
					train_f.copyBlock(test_row * k, 0, f, test_row + test_row * k, 0, train_row - test_row * k, feat_cols);
					train_l.copyBlock(test_row * k, 0, l, test_row + test_row * k, 0, train_row - test_row * k, labels_cols);
					
				}
				train(train_f, train_l);				
				sse += measureSSE(fi, li);
			}
		}
		return sse / reps;
	}
}