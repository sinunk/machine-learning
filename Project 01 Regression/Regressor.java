import java.lang.*;

public class Regressor extends SupervisedLearner
{

	
	String name()
	{
		return "Regression Learner";
	}
	
	double[] params;
	Function model;
	double[] input;
	double[] allParams;

	public Regressor(Function m) {
		model = m;
	}

	public void setParams(double[] p) {
	if (p==null)
		throw new RuntimeException("p should not be null");
		params = p;
	}
	
	//in = X, params = m,b
	public void predict(double[] in, double[] out) {
	if (in==null)
		throw new RuntimeException("in should not be null");
		input = in;
		allParams = Vec.concatenate(params, input);
		out[0] = model.evaluate(allParams);
	}
	
	public void train(Matrix features, Matrix labels) {
		if(labels.cols() != 1)
			throw new RuntimeException(
				"Expected labels to have only one column");
		Objective ob = new Objective(features, labels, this);
		HillClimber hc = new HillClimber(features.cols()+1, features.cols()+1, ob);
		hc.optimize(200, 50, 0.01);
	}

}
