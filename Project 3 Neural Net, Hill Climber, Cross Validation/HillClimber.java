import java.lang.*;
import java.util.*;

//public class HillClimber extends Optimizer
public class HillClimber
{
	double[] cur;
	double[] step;
	double smallest_error;
	double[] error = new double[5];
	double[] m_weights_vector;
	NeuralNet m_nn;
	Matrix features, labels;
	
	double getMinimum(double[] array)
	{
		double minValue = array[0];
		for(int i = 0; i < array.length; i++)
		{
			if(array[i] < minValue)
				{
					minValue = array[i];
				}
		} 
		return minValue;
	}
	
	//HillClimber(int curSize, int stepSize, Objective obj, NeuralNet nn)
    HillClimber(Matrix feat, Matrix lab, NeuralNet nn)
	{	
		//super(obj);
		/*
		//initialize with random values
		Random rand = new Random();
		cur = new double[curSize];
		for(int i = 0; i < curSize; i++)
		{
            cur[i] = 0.03 * rand.nextGaussian();
        }
        */
		/*
		//unittest
		cur[0] = 0.1;
		cur[1] = 0.1;
		cur[2] = 0.1;
		cur[3] = 0.3;
		cur[4] = 0.1;
		cur[5] = -0.1;
		cur[6] = 0.1;
		cur[7] = -0.2;
		*/

		m_nn = nn;
		features = feat;
		labels = lab;

		//prepare to reuse random seed
		//double[] weights_random_seed = {0.044988552089481895,-0.0020598350461662617,-0.02992832764507834,0.0017637216282844243,-0.028812033865045036,-0.011074350225623069,0.007831600296341374,0.042568095215429466,-0.020412478698313242,-0.054790973214575364,-0.039414128974615416,-0.03267355152121993,0.017587061845507488,0.015931790325061215,0.029420063499723505,0.042816978870060425,-0.004469847596846707,0.024364208041895864,0.008722272014040662,-0.01099057531726095,-0.013140925549742674,-0.01087522212110316,0.01323939709627569,0.008316932895686198,-0.04227620441256218,-0.030815392118150972,0.014117032259828265,0.00933836535427071,-0.025336353692512564,0.062118274902278545,-0.017736640964552363,-0.06165248353786871,-0.00937335229595439,6.465817704972589E-4,0.03829059051538035,-0.02542394714137862,-0.020593780604898825,-0.014881618202228058,0.026284003056574433,0.029571883955570564,-0.004984982191658225,0.029035781148265036,-0.02894534928169949,0.03344609578119433,0.006199487634878217,0.023569891812784356,0.017992555492150272,-0.021310910158353423,0.013814695376087581,-0.026766957678728928,0.022604630839925383,0.07485739083478248,-0.020405634020001586,0.006998015452606554,0.00793719464866904,-0.05111507766449735,0.019121780580202176,-0.060670536428469105,0.018575492443717395,0.018659547290687622,-0.009103163671122568,0.00613201615959993,-0.044388469228208174,-0.0335478936047761,-0.01696985606330438,0.011750333479544673,-0.028117950129343254,-0.008720027285568496,-0.004102374319246911,-0.023687110829050586,-0.040606030848009386,0.0940848706050732,0.008537079573862443,-0.004843144593719511,0.0077616181099204136,-0.03210877666343537,0.07643666516700293,0.03840930532767629,-0.04054833964082936,-0.006169861063000116,0.003851606104431388,-0.032184856898295894,0.04413291005583165,0.024639181895263665,0.0019967955695111544,0.0393609618607917,0.01627634428245781,0.025279942978523356,0.03343513096343554,-0.0026412014566965735,-0.03396439517438049,-0.038090838602763015,0.006968750796460016,0.03022405573433462,-0.045818428889043486,-0.014948882525031631,-0.0695653000706814,0.020877733014708192,-0.018000736530824832,0.0032484480612230884,-6.8601924913101E-4,-0.044741176796873415,-0.0019186960548741297,-0.008968560967403722,0.041823239938507326,-0.017789527050941348,0.060404807850042985,0.01926397886528959,0.044946438222953425,0.026098507254450305,9.327531584463107E-4,0.025090483062385077,-0.010858227751162663,-0.014131555602167613,-0.008684807918506107,-0.0459505745432053,0.060511334550754275,0.013407948707477734,0.0027258273659229648,0.06255305748922237,0.015083290645061944};
		m_nn.init(); //initialize with small random values
        m_weights_vector = m_nn.getWeightsVector();
        //take weights from nn, convert to Vec, copy to cur
        cur = new double[m_weights_vector.length];
        Vec.copy(cur, m_weights_vector);

        //saves random seed
		//System.out.print("initial cur");
		//Vec.println(cur);
		//System.out.print("cur size:" + curSize);
		
		step = new double[m_weights_vector.length];
		for(int i = 0; i < m_weights_vector.length; i++)
			step[i] = 0.1;
	
	}

	double iterate()
	{
		if (cur == null)
			throw new RuntimeException("cur should not be null");
		//System.out.print("Cur Before: ");
        //Vec.println(cur);
		
		for(int i = 0; i < cur.length; i++)
		{
			error[0] = evaluate(cur);
			cur[i] = cur[i] - 1.25 * step[i];
			error[1] = evaluate(cur);
			cur[i] = cur[i] + 1.25 * step[i];
			cur[i] = cur[i] - 0.8 * step[i];
			error[2] = evaluate(cur);
			cur[i] = cur[i] + 0.8 * step[i];
			cur[i] = cur[i] + 0.8 * step[i];
			error[3] = evaluate(cur);
			cur[i] = cur[i] - 0.8 * step[i];
			cur[i] = cur[i] + 1.25 * step[i];
			error[4] = evaluate(cur);
			cur[i] = cur[i] - 1.25 * step[i];
		
			smallest_error = getMinimum(error);
		
			if(smallest_error == error[0])
				step[i] = 0.8 * step[i];

			//else if(smallest_error == error[1])
			if(smallest_error == error[1])
			{
				cur[i] = cur[i] - 1.25 * step[i];
				step[i] = 1.25 * step[i];
			}
			//else if(smallest_error == error[2])
			if(smallest_error == error[2])
			{
				cur[i] = cur[i] - 0.8 * step[i];
				step[i] = 0.8 * step[i];
			}
            //else if(smallest_error == error[3])
			if(smallest_error == error[3])
			{
				cur[i] = cur[i] + 0.8 * step[i];
				step[i] = 0.8 * step[i];
			}
            //else if(smallest_error == error[4])
			if(smallest_error == error[4])
			{
				cur[i] = cur[i] + 1.25 * step[i];
				step[i] = 1.25 * step[i];
			}
		}
		
		//System.out.print("Cur After: ");
        //Vec.println(cur);
		return smallest_error;
	}

    double evaluate(double[] cur)
    {
        double sse = 0.0;
        m_nn.setWeightsBiases(cur);
        sse = m_nn.measureSSE(features, labels);
        //System.out.println(sse);
        return sse;
    }

    double optimize(int burnIn, int window, double thresh)
    {	double error = 0.0;

        for(int i = 1; i < burnIn; i++)
        {
            iterate();
            error = iterate();

            while(true)
            {
                double prevError = error;

                for(int j = 1; j < window; j++)
                    iterate();

                error = iterate();

                if((prevError - error) / prevError < thresh || error == 0.0)
                    break;
            }
        }
        return error;
    }
}