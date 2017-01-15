/// Fundamental interface for all unsupervised learning algorithms.
abstract public class UnsupervisedLearner 
{
	/// Trains this unsupervised learner
	abstract void train(Matrix data);
	
	/// Returns an example of an output matrix. The meta-data of this matrix
	///shows how output will be given. (This matrix contains no data because it has zero rows.
	///It is only used for the meta-data.
	abstract Matrix outputTemplate();
	
	/// Transform a single instance
	abstract void transform(double[] in, double[] out);
	
	/// Untransform a single instance
	abstract void untransform(double[] in, double[] out);
}
