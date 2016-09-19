public class Objective extends Function
{
	/*
	The constructor for the Objective class should require 3 parameters: 
	a matrix named features, a matrix named labels, and a Regressor. 
	The evaluate method should pass the vector to the setParams method of the Regressor object, 
	then call measureSSE to measure the sum-squared-error with the features and labels. 
	(The purpose of this function is to evaluate a vector. 
	If it returns a big number, that means the vector is poor. 
	If it returns a small number, that means the vector is good.)
	*/
	
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