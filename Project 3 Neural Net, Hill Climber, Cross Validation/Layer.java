import java.lang.*;

public class Layer
{
	Matrix m_weights = new Matrix();
	double[] m_bias;
	double[] m_activation;
	double[] m_weights_vector;
	double[] m_net;

	Layer(int inputs, int outputs)
	{
		if(inputs < 0 || outputs < 0)
			throw new RuntimeException("inputs or outputs can't be negative!");
		m_weights.setSize(outputs, inputs);
		m_bias = new double[outputs];
		m_activation = new double[outputs];
		m_net = new double[m_activation.length];
		m_weights_vector = new double[m_weights.rows() * (m_weights.cols()+1)];
	}
	
	//compute the activation
	void feedForward(double[] x)
	{
		//compute the net element-wise

		//debugging
		/*
		System.out.println("weights.cols:" + weights.cols());
		System.out.println("weights.rows:" + weights.rows());
		System.out.println("x.length:" + x.length);
		*/

		for(int i = 0; i < m_weights.rows(); i++)
		{
			m_net[i] += Vec.dotProduct(m_weights.row(i), x);
		}
		Vec.add(m_net, m_bias);
		/*
        //debugging
        System.out.println("weights:");
        for(int i = 0; i < m_weights.rows(); i++)
        {
            Vec.println(m_weights.row(i));
        }
        //System.out.println("bias:");
        //Vec.println(m_bias);
        //System.out.println("net:");
        //Vec.println(m_net);
        */
		//compute activation
		for(int k = 0; k < m_activation.length; k++)
		{
			m_activation[k] = Math.tanh(m_net[k]);
            //debugging
            //System.out.println("activation:");
            //Vec.println(m_activation);
		}
	}
	
	double[] getActivation()
	{
		return m_activation;
	}

	//check indexing on getWeights and setWeights
    //problem here
	double[] getWeightsVector(int pos)
    {
        for (int x = 0; x < m_weights.rows(); x++)
        {
            for (int y = 0; y < m_weights.cols(); y++)
            {
                m_weights_vector[pos] = m_weights.row(x)[y];
                pos++;
            }
        }

        for (int z = 0; z < m_bias.length; z++)
        {
            m_weights_vector[pos] = m_bias[z];
            pos++;
        }

        return m_weights_vector;
    }

	//copyBlock(int destRow, int destCol, Matrix that, int rowBegin, int colBegin, int rowCount, int colCount)
	void setWeights(Matrix w)
	{
		m_weights.copyBlock(0, 0, w, 0, 0, w.rows(), w.cols());
	}
	
	void setBiases(double[] b)
	{
		Vec.copy(m_bias, b);
	}
}