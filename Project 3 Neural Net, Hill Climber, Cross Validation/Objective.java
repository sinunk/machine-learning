public class Objective extends Function
{	
	Matrix features, labels;
	NeuralNet net;
	
	Objective (Matrix feat, Matrix lab, NeuralNet nn) 
	{
		features = feat;
		labels = lab;
		net = nn;
	}
	
	double evaluate(double[] in)
	{	
		double sse = 0.0;
		net.setWeightsBiases(in);
		sse = net.measureSSE(features, labels);
		//System.out.println(sse);
		return sse;
	}
}