// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

/// This class wraps another supervised learner. It applies some unsupervised
/// operation to the data before presenting it to the learner.
public class Filter extends SupervisedLearner
{
	private SupervisedLearner m_pLearner;
	private UnsupervisedLearner m_pTransform;
	private boolean m_filterInputs;
	private double[] m_buffer;

	String name() { return "Filter"; }
	
	/// This takes ownership of pLearner and pTransform.
	/// If inputs is true, then it applies the transform only to the input features.
	/// If inputs is false, then it applies the transform only to the output labels.
	/// (If you wish to transform both inputs and outputs, you must wrap a filter in a filter)
	public Filter(SupervisedLearner pLearner, UnsupervisedLearner pTransform, boolean filterInputs)
	{
		m_pLearner     = pLearner;
		m_pTransform   = pTransform;
		m_filterInputs = filterInputs;
	}
	
	/// Train the transform and the inner learner
	public void train(Matrix features, Matrix labels)
	{
		if (features.rows() != labels.rows())
			throw new RuntimeException("Expected features and labels to have the same number of rows");

		if (m_filterInputs)
		{
			m_pTransform.train(features);
			m_buffer = new double[m_pTransform.outputTemplate().cols()];
			Matrix temp = new Matrix();
			temp.copyMetaData(m_pTransform.outputTemplate());
			temp.newRows(features.rows());
			for (int i = 0; i < features.rows(); i++)
				m_pTransform.transform(features.row(i), temp.row(i));
			m_pLearner.train(temp, labels);
		}
		else
		{
			m_pTransform.train(labels);
			m_buffer = new double[m_pTransform.outputTemplate().cols()];
			Matrix temp = new Matrix();
			temp.copyMetaData(m_pTransform.outputTemplate());
			temp.newRows(labels.rows());
			for (int i = 0; i < labels.rows(); i++)
				m_pTransform.transform(labels.row(i), temp.row(i));
			m_pLearner.train(features, temp);
		}
	}
	/*
	/// Train the transform and the inner learner per Epoch
	public void trainEpoch(Matrix features, Matrix labels)
	{
		if (features.rows() != labels.rows())
			throw new RuntimeException("Expected features and labels to have the same number of rows");

		if (m_filterInputs)
		{
			m_pTransform.train(features);
			m_buffer = new double[m_pTransform.outputTemplate().cols()];
			Matrix temp = new Matrix();
			temp.copyMetaData(m_pTransform.outputTemplate());
			temp.newRows(features.rows());
			for (int i = 0; i < features.rows(); i++)
				m_pTransform.transform(features.row(i), temp.row(i));
			m_pLearner.trainEpoch(temp, labels);
		}
		else
		{
			m_pTransform.train(labels);
			m_buffer = new double[m_pTransform.outputTemplate().cols()];
			Matrix temp = new Matrix();
			temp.copyMetaData(m_pTransform.outputTemplate());
			temp.newRows(labels.rows());
			for (int i = 0; i < labels.rows(); i++)
				m_pTransform.transform(labels.row(i), temp.row(i));
			m_pLearner.trainEpoch(features, temp);
		}
	}
	*/
	/// Make a prediction
	public void predict(double[] in, double[] out)
	{
		if (m_filterInputs)
		{
			m_pTransform.transform(in, m_buffer);
			m_pLearner.predict(m_buffer, out);
		}
		else
		{
			m_pLearner.predict(in, m_buffer);
			m_pTransform.untransform(m_buffer, out);
		}
	}
}
