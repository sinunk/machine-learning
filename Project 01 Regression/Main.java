class Main
{
	static void testLinear(SupervisedLearner learnerLinear, String challenge)
	{
		// Load the data
		String fn = "data/" + challenge;
		Matrix features = new Matrix();
		features.loadARFF(fn + "_features.arff");
		Matrix labels = new Matrix();
		labels.loadARFF(fn + "_labels.arff");

		// Train the model
		learnerLinear.train(features, labels);

		// Measure and report accuracy
		double sse_linear = learnerLinear.measureSSE(features, labels);
		System.out.println("sse_linear=" + Double.toString(sse_linear));	

	}

	public static void testLearnerLinear(SupervisedLearner learnerLinear)
	{
		testLinear(learnerLinear, "linear");
	}
	
	static void testParabola(SupervisedLearner learnerParabola, String challenge)
	{
		// Load the data
		String fn = "data/" + challenge;
		Matrix features = new Matrix();
		features.loadARFF(fn + "_features.arff");
		Matrix labels = new Matrix();
		labels.loadARFF(fn + "_labels.arff");

		// Train the model
		learnerParabola.train(features, labels);

		// Measure and report accuracy
		double sse_parabola = learnerParabola.measureSSE(features, labels);
		System.out.println("sse_parabola=" + Double.toString(sse_parabola));	

	}

	public static void testLearnerParabola(SupervisedLearner learnerParabola)
	{
		testParabola(learnerParabola, "linear");
	}
	
	static void testHousing(SupervisedLearner learnerHousing, String challenge)
	{
		// Load the data
		String fn = "data/" + challenge;
		Matrix features = new Matrix();
		features.loadARFF(fn + "_features.arff");
		Matrix labels = new Matrix();
		labels.loadARFF(fn + "_labels.arff");

		// Train the model
		learnerHousing.train(features, labels);

		// Measure and report accuracy
		double sse_housing = learnerHousing.measureSSE(features, labels);
		System.out.println("sse_housing=" + Double.toString(sse_housing));	
	}

	public static void testLearnerHousing(SupervisedLearner learnerHousing)
	{
		testHousing(learnerHousing, "housing");
	}

	public static void main(String[] args)
	{
		testLearnerLinear(new Regressor(new ModelLinear()));
		testLearnerParabola(new Regressor(new ModelParabola()));
		testLearnerHousing(new Regressor(new ModelHousing()));
	}
}
