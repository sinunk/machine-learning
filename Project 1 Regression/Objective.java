public class Objective extends Function
{
	Matrix features, labels;
	Regressor reg;
	
	Objective (Matrix feat, Matrix lab, Regressor regress) 
	{
		features = feat;
		labels = lab;
		reg = regress;
	}
	
	double evaluate(double[] in)
	{	
		double sse = 0.0;
		reg.setParams(in);
		sse = reg.measureSSE(features, labels);
		return sse;
	}
}
