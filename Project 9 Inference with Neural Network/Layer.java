import java.lang.*;


public class Layer
{
    Matrix m_weights = new Matrix(); // cols = in, rows = out
    double[] m_bias;
    double[] m_net; // net = in * weights + bias
    double[] m_activation; // activation = tanh(net)
    double[] m_blame;
    Matrix m_weightsDelta = new Matrix(); // how much to change each weight
    double[] m_biasDelta; // how much to change each bias value

    Layer(int inputs, int outputs)
    {
        if(inputs < 0 || outputs < 0)
            throw new RuntimeException("inputs or outputs can't be negative!");
        if (m_weights == null)
            throw new RuntimeException("m_weights should not be null");
        m_weights.setSize(outputs, inputs);
        m_bias = new double[outputs];
        m_net = new double[outputs];
        m_activation = new double[outputs];
        m_blame = new double[outputs];
        m_biasDelta = new double[outputs];
        m_weightsDelta.setSize(outputs, inputs);
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

    //add support for arbitrary activation function
    double atan (double input)
    {
        return Math.atan(input);
    }

    //regular tanh
    double tanh (double input)
    {
        return Math.tanh(input);
    }

    /*
    //yan le cun, efficient backprop paper
    double tanh (double input)
    {
        return 1.7519 * Math.tanh((2 / 3) * input);
    }
    */
    double sigmoid(double input)
    {
        return (1 / (1 + Math.pow (Math.E,(- 1.0 * input))));
    }

    double sigmoid_prime(double input)
    {
        return (sigmoid(input) * (1.0 - sigmoid(input)));
    }

    double atan_prime (double input)
    {
        return 1.0 / (input * input + 1.0);
    }

    double tanh_prime (double input)
    {
        return 1.0 - (tanh(input) * tanh(input));
    }

    //identity input range: -infinity, infinity
    double identity (double input) { return input;}
    double identity_prime (double input) { return 1.0;}

    /// called by present_pattern: feed forward, get blame, backprop, update delta
    //compute the activation = tanh(in * weights + bias)
    void feedForward(double[] x)
    {
        //compute tanh input element-wise
        //double[] m_net = new double[m_activation.length];
        Vec.copy(m_net, m_bias);
        //debugging
        /*
		System.out.println("weights.cols:" + m_weights.cols());
		System.out.println("weights.rows:" + m_weights.rows());
		System.out.println("x.length:" + x.length);
        */
        for(int i = 0; i < m_weights.rows(); i++)
        {
            m_net[i] += Vec.dotProduct(m_weights.row(i), x);
        }
        //Vec.add(m_net, m_bias);

        //compute activation
        for(int k = 0; k < m_activation.length; k++)
        {
            //m_activation[k] = sigmoid(m_net[k]); // sigmoid
            m_activation[k] = tanh(m_net[k]); // tanh
            //m_activation[k] = atan(m_net[k]); // atan
            //m_activation[k] = identity(m_net[k]); // identity
        }
        /*
        //debugging
        System.out.println("m_weights:");
        m_weights.print();
        System.out.println("m_bias:");
        Vec.println(m_bias);
        System.out.println("m_net:");
        Vec.println(m_net);
        System.out.println("m_activation:");
        Vec.println(m_activation);

        System.out.println("m_blame:");
        Vec.println(m_blame);
        System.out.println("m_weightsDelta:");
        m_weightsDelta.print();
        System.out.println("m_biasDelta:");
        Vec.println(m_biasDelta);
        */
    }

    double[] getActivation()
    {
        return m_activation;
    }

    double[] getNet()
    {
        return m_net;
    }

    ///done
    double[] getOutputBlame(double[] target)
    {
        //blame = (target - prediction) * a'(net)
        //target = labels, prediction = a(net)
        //element-wise!
        if (target == null)
            throw new RuntimeException("target in getOutputBlame should not be null");
        if (m_blame == null)
            throw new RuntimeException("m_blame in getOutputBlame should not be null");
        if (m_activation == null)
            throw new RuntimeException("m_activation in getOutputBlame should not be null");
        if (m_net == null)
            throw new RuntimeException("m_net in getOutputBlame should not be null");

        for(int i = 0; i < target.length; i++)
        {
            //m_blame[i] = (target[i] - m_activation[i]) * sigmoid_prime(m_net[i]); // sigmoid
            m_blame[i] = (target[i] - m_activation[i]) * tanh_prime(m_net[i]); // tanh
            //m_blame[i] = (target[i] - m_activation[i]) * atan_prime(m_net[i]); //atan
            //m_blame[i] = (target[i] - m_activation[i]) * identity_prime(m_net[i]); //identity
        }

        return m_blame;
    }

    double[] getBlame()
    {
        return m_blame;
    }
    /*
	/// Computes blame = from.blame * from.weights * tanh'(net)
    //element-wise!
	//void updateBlame(Layer from, double[] from_blame)
    void updateBlame(Layer from)
	{
	    double[] from_net = from.getNet();
	    double[] from_blame = from.getBlame();


        //debugging
        System.out.println("m_weights.rows()");
        System.out.println(m_weights.rows());
        System.out.println("m_weights.cols()");
        System.out.println(m_weights.cols());
        System.out.println("m_blame.length");
        System.out.println(m_blame.length);
        System.out.println("from_net.length");
        System.out.println(from_net.length);

		for(int i = 0; i < m_weights.rows(); i++)
		{
			double blame = 0.0;
			for (int j = 0; j < from.m_weights.rows(); j++)
			{
				blame += from.m_weights.row(j)[i] * from_blame[j]
                        * tanh_prime(from_net[i]);
			}
			m_blame[i] = blame;
		}
	}
    */
    void backpropBlame(Layer from)
    {
//        double[] from_net = from.getNet();
        //double[] from_blame = from.getBlame();

        /*
        //debugging
        System.out.println("m_weights.rows()");
        System.out.println(m_weights.rows());
        System.out.println("m_weights.cols()");
        System.out.println(m_weights.cols());
        System.out.println("m_blame.length");
        System.out.println(m_blame.length);
        System.out.println("from_net.length");
        System.out.println(from_net.length);
        System.out.println("from_blame.length");
        System.out.println(from_blame.length);
        */
        //m_blame = 20
        //m_weights = 20 x 27
        //from_net = 11
        //from_blame = 11
        for(int i = 0; i < m_weights.rows(); i++)
        {
            double blame = 0.0;
            for (int j = 0; j < from.m_weights.rows(); j++)
            {
//                blame += from.m_weights.row(j)[i] * from_blame[j]
//                        * tanh_prime(from_net[i]);

                //blame += from.m_weights.row(j)[i] * from.m_blame[j] * sigmoid_prime(m_net[i]); //ok sigmoid
                blame += from.m_weights.row(j)[i] * from.m_blame[j] * tanh_prime(m_net[i]); //ok tanh
                //blame += from.m_weights.row(j)[i] * from.m_blame[j] * atan_prime(m_net[i]); //atan
                //blame += from.m_weights.row(j)[i] * from.m_blame[j] * identity_prime(m_net[i]); //identity
            }
            m_blame[i] = blame;
        }
    }

    /*
    Except we subtract (lambda * learning_rate * weight)from each weight
    just before we update the weights in the usual manner.
    And that is the same as multiplying all the weights
    in the network by (1 - (learning_rate * lambda))
    just before updating the weights in the usual manner.
    */
    //Implement L1 Regularization
    void regularize_L1(double l1, double learning_rate)
    {
        double lambda1 = l1;
        for(int i = 0; i < m_weights.rows(); i++)
        {
            double[] w = m_weights.row(i);

            for(int j = 0; j < m_weights.cols(); j++)
                w[j] *= (1 - (lambda1 * learning_rate));

            m_bias[i] *= (1 - (lambda1 * learning_rate));
        }
    }

    /*
    It turns out to be to subtract (lambda * learning_rate) from all the positive weights,
    and to add (lambda * learning_rate) to all the negative weights,
    just before you update those weights in the usual manner.
    */
    //Implement L2 Regularization
    void regularize_L2(double l2, double learning_rate)
    {
        double lambda2 = l2;
        for(int i = 0; i < m_weights.rows(); i++)
        {
            double[] w = m_weights.row(i);
            for(int j = 0; j < m_weights.cols(); j++)
                if (w[j] < 0)
                    w[j] += (lambda2 * learning_rate);
                else
                    w[j] -= (lambda2 * learning_rate);

            if (m_bias[i] < 0)
                m_bias[i] += (lambda2 * learning_rate);
            else
                m_bias[i] -= (lambda2 * learning_rate);
        }
    }

    /// delta += outer_product(blame, in); done
    void updateDeltas(double[] in)
    {
        for(int j = 0; j < m_weights.rows(); j++)
        {
            //double[] wd = m_weightsDelta.row(j);
            for(int i = 0; i < m_weights.cols(); i++)
            {
                //m_weightsDelta.row(j)[i] = in[i] * m_blame[j];
                m_weightsDelta.row(j)[i] += in[i] * m_blame[j];
            }
            //m_biasDelta[j] = m_blame[j];
            m_biasDelta[j] += m_blame[j];
        }
        /*
        //debugging
        System.out.println("m_weights:");
        m_weights.print();
        System.out.println("m_bias:");
        Vec.println(m_bias);
        System.out.println("m_net:");
        Vec.println(m_net);
        System.out.println("m_activation:");
        Vec.println(m_activation);

        System.out.println("m_blame:");
        Vec.println(m_blame);
        System.out.println("m_weightsDelta:");
        m_weightsDelta.print();
        System.out.println("m_biasDelta:");
        Vec.println(m_biasDelta);
        */
    }

    /// called by present_pattern: feed forward, get blame, backprop, update delta
    /// called by train: L.decay delta, present pattern, L.update weights
    /// delta *= momentum; done
    void decayDeltas(double momentum)
    {
        //element-wise
        for(int i = 0; i < m_weightsDelta.rows(); i++)
        {
            Vec.scale(m_weightsDelta.row(i), momentum);
        }
        Vec.scale(m_biasDelta, momentum);
    }

    /// weights += learning_rate * delta; done
    void updateWeights(double learning_rate)
    {
        for(int i = 0; i < m_weights.rows(); i++)
        {
            //double[] w = m_weights.row(i);
            //double[] wd = m_weightsDelta.row(i);
            for(int j = 0; j < m_weights.cols(); j++)
            {
                m_weights.row(i)[j] += (learning_rate * m_weightsDelta.row(i)[j]);
            }
            m_bias[i] += learning_rate * m_biasDelta[i];
        }
    }
}