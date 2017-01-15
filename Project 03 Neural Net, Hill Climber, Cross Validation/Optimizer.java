abstract public class Optimizer 
{

	Objective obj;

	Optimizer(Objective ob)
	{
		obj = ob;
	}
	
	abstract double iterate ();

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