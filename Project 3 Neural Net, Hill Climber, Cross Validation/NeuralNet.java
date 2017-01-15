import java.util.ArrayList;

//Contains a collection of Layer objects that feed into each other.
public class NeuralNet extends SupervisedLearner
{
	ArrayList<Layer> layers = new ArrayList<Layer>();
	int[] layer_sizes;
	Matrix m_w;
	double[] m_b;
	double[] m_allWeightsVector;

	String name()
	{
		return "Neural Network";
	}

    NeuralNet(int[] layerSizes)
    {
    	layer_sizes = layerSizes;
    	int allWB_size = 0;

    	for(int i = 1; i < layer_sizes.length; i++)
        {
            layers.add(new Layer(layer_sizes[i - 1], layer_sizes[i]));
            allWB_size += (layer_sizes[i] * layer_sizes[i - 1] + layer_sizes[i]);
            //i = 1, Layer(13, 8)
            //i = 2, Layer(8, 1)

        }
        /*
        for (int i = 0; i < layer_sizes.length - 1; i++)
        {
            m_w = new Matrix(layer_sizes[i + 1], layer_sizes[i]);
            m_b = new double[layer_sizes[i + 1]];
        }
        */
        m_allWeightsVector = new double[allWB_size];
    }
    
	//feed "in" to first layer and so on
	void predict(double[] in, double[] out)
	{
		if(in == null)
			throw new RuntimeException("in can't be null");
		layers.get(0).feedForward(in);
		//j = 1, Layer(13, 8)
		//j = 2, Layer(8, 1)
		for(int j = 1; j < layers.size(); j++)
		{	
			layers.get(j).feedForward(layers.get(j-1).getActivation());
        }
		Vec.copy(out, layers.get(layers.size() - 1).getActivation());
        //debugging
        //System.out.println("predict output:");
        //Vec.println(out);
        //System.out.println("predict input:");
        //Vec.println(in);
	}

	//initialize weights
    //prepare to reuse random seed
    void init()
    {
        //13->8->1 : (13*8+8) + (8*1+1) = 121
        double[] weights_random_seed = {0.044988552089481895, -0.0020598350461662617, -0.02992832764507834, 0.0017637216282844243, -0.028812033865045036, -0.011074350225623069, 0.007831600296341374, 0.042568095215429466, -0.020412478698313242, -0.054790973214575364, -0.039414128974615416, -0.03267355152121993, 0.017587061845507488, 0.015931790325061215, 0.029420063499723505, 0.042816978870060425, -0.004469847596846707, 0.024364208041895864, 0.008722272014040662, -0.01099057531726095, -0.013140925549742674, -0.01087522212110316, 0.01323939709627569, 0.008316932895686198, -0.04227620441256218, -0.030815392118150972, 0.014117032259828265, 0.00933836535427071, -0.025336353692512564, 0.062118274902278545, -0.017736640964552363, -0.06165248353786871, -0.00937335229595439, 6.465817704972589E-4, 0.03829059051538035, -0.02542394714137862, -0.020593780604898825, -0.014881618202228058, 0.026284003056574433, 0.029571883955570564, -0.004984982191658225, 0.029035781148265036, -0.02894534928169949, 0.03344609578119433, 0.006199487634878217, 0.023569891812784356, 0.017992555492150272, -0.021310910158353423, 0.013814695376087581, -0.026766957678728928, 0.022604630839925383, 0.07485739083478248, -0.020405634020001586, 0.006998015452606554, 0.00793719464866904, -0.05111507766449735, 0.019121780580202176, -0.060670536428469105, 0.018575492443717395, 0.018659547290687622, -0.009103163671122568, 0.00613201615959993, -0.044388469228208174, -0.0335478936047761, -0.01696985606330438, 0.011750333479544673, -0.028117950129343254, -0.008720027285568496, -0.004102374319246911, -0.023687110829050586, -0.040606030848009386, 0.0940848706050732, 0.008537079573862443, -0.004843144593719511, 0.0077616181099204136, -0.03210877666343537, 0.07643666516700293, 0.03840930532767629, -0.04054833964082936, -0.006169861063000116, 0.003851606104431388, -0.032184856898295894, 0.04413291005583165, 0.024639181895263665, 0.0019967955695111544, 0.0393609618607917, 0.01627634428245781, 0.025279942978523356, 0.03343513096343554, -0.0026412014566965735, -0.03396439517438049, -0.038090838602763015, 0.006968750796460016, 0.03022405573433462, -0.045818428889043486, -0.014948882525031631, -0.0695653000706814, 0.020877733014708192, -0.018000736530824832, 0.0032484480612230884, -6.8601924913101E-4, -0.044741176796873415, -0.0019186960548741297, -0.008968560967403722, 0.041823239938507326, -0.017789527050941348, 0.060404807850042985, 0.01926397886528959, 0.044946438222953425, 0.026098507254450305, 9.327531584463107E-4, 0.025090483062385077, -0.010858227751162663, -0.014131555602167613, -0.008684807918506107, -0.0459505745432053, 0.060511334550754275, 0.013407948707477734, 0.0027258273659229648, 0.06255305748922237, 0.015083290645061944};
        //double[] weights_random_seed = {0.0, 0.0}; //debugging
        //wv[i] = 0.03 * rand.nextGaussian();
        //wv = new double[weights_random_seed.length];
        //Vec.copy(wv, weights_random_seed);
        //System.out.println("weights_random_seed.length");
        //System.out.println(weights_random_seed.length);
        setWeightsBiases(weights_random_seed);
    }

	//set all weights and biases from a big vector that will be optimized by HC
	void setWeightsBiases(double[] wb)
	{
		if(wb == null)
			throw new RuntimeException("wb should not be null");

		int idx = 0;
		for (int i = 0; i < layer_sizes.length - 1; i++)
		{
			//i = 1, Layer(13, 8)
			//i = 2, Layer(8, 1)
            m_w = new Matrix(layer_sizes[i + 1], layer_sizes[i]);
            m_b = new double[layer_sizes[i + 1]];

			for (int x = 0; x < m_w.rows(); x++)
			{
				for (int y = 0; y < m_w.cols(); y++)
				{
					m_w.row(x)[y] = wb[idx];
					idx++;
				}
			}
			
			for (int z = 0; z < m_b.length; z++)
			{
				m_b[z] = wb[idx];
				idx++;
			}
			
			layers.get(i).setWeights(m_w);
			layers.get(i).setBiases(m_b);
		}
		//Vec.println(wb);
	}
	
	//use HillClimber to optimize weights and bias	
	void trainEpoch(Matrix features, Matrix labels)
	{
		//init();
		//call HC just once.
		if(labels.cols() != 1)
			throw new RuntimeException("Expected labels should only have one column");
		//Objective ob = new Objective(features, labels, this);
		HillClimber hc = new HillClimber(features, labels, this);
		hc.iterate();
	}

    //concatenate all weights vectors from all layers
    //problem here
    double[] getWeightsVector()
    {
        int pos = 0;
        int idx = 0;
        for (int i = 0; i < layer_sizes.length - 1; i++)
        {
            double[] layer_vector = layers.get(i).getWeightsVector(pos);
            //i = 0,
            //pos += layer_vector.length;
			//m_allWeightsVector = Vec.concatenate(m_allWeightsVector, layer_vector);
            for(int j = 0; j < layer_vector.length; j++)
            {
                m_allWeightsVector[idx] = layer_vector[j];
                idx++;
            }
        }
        return m_allWeightsVector;
    }
	
	//use HillClimber to optimize weights and bias
	void train(Matrix features, Matrix labels)
	{
		if(labels.cols() != 1)
			throw new RuntimeException("Expected labels should only have one column");
		//if(allWeightsBiases == null)
		//	throw new RuntimeException("allWeightsBiases should not be null");

        //send concatenated weights vectors to hill climber
        //this is done in hill climber constructor

		//Objective ob = new Objective(features, labels, this);
		//HillClimber hc = new HillClimber(allWeightsBiases.length, allWeightsBiases.length, ob, this);
        //init();
		HillClimber hc = new HillClimber(features, labels, this);
        hc.optimize(200, 50, 0.001);

		//send updated weights from hill climber to all layers back
        //this is done in hill climber evaluate

	}
	
}