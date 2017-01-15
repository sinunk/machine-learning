abstract class SupervisedLearner 
{
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels);

	/// Make a prediction
	abstract void predict(double[] in, double[] out);

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
			/*
			System.out.print("feat=");
			Vec.print(feat);
			System.out.print(", lab=");
			Vec.print(lab);
			System.out.print(", pred=");
			Vec.print(pred);
			System.out.print(", sse=");
			System.out.print(sse);
			*/
		}
	return sse;
	}
}
