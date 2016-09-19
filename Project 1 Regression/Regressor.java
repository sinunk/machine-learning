import java.lang.*;

public class Regressor extends SupervisedLearner
{
/*
Modify the constructor of your Regressor class to require a Function to be passed as a parameter. 
Store a reference to this Function in a member variable named model. 
(As always, you are welcome to do things your own way, as long as your way works.) 

Implement the predict method to concatenate the parameters, params, with the variables, in, 
and call the evaluate method of model. (Note that model is not the objective function.) 

Implement the train method to regress the parameters of the model to fit with features and labels. 
To do this, instantiate one of your optimizers and call optimize. 
Here is some Java-style example pseudocode:
*/
	
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
/*
(For example, if your model is "y=mx+b", then "x" is a variable or feature, and "m" and "b" are parameters. 
The HillClimber will visit many different points in the parameter-space to find the one that fits well with the data.)
*/
}