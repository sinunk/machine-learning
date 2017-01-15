public class HillClimber extends Optimizer 
{
	double[] cur;
	double[] step;
	double smallest_error;
	double[] error = new double[5];
	
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

	HillClimber(int curSize, int stepSize, Objective obj)
	{	
		super(obj);
		
		cur = new double[curSize];
		for(int i = 0; i < curSize; i++)
			cur[i] = 0.0;
		
		step = new double[stepSize];			
		for(int i = 0; i < stepSize; i++)
			step[i] = 0.1;
	
	}

	double iterate()
	{
		if (cur==null)
		throw new RuntimeException("cur should not be null");

		for(int i = 0; i < cur.length; i++)
		{
			error[0] = obj.evaluate(cur);
			cur[i] = cur[i] - 1.25 * step[i];
			error[1] = obj.evaluate(cur);
			cur[i] = cur[i] + 1.25 * step[i];
			cur[i] = cur[i] - 0.73 * step[i];
			error[2] = obj.evaluate(cur);
			cur[i] = cur[i] + 0.73 * step[i];
			cur[i] = cur[i] + 0.73 * step[i];
			error[3] = obj.evaluate(cur);
			cur[i] = cur[i] - 0.73 * step[i];
			cur[i] = cur[i] + 1.25 * step[i];
			error[4] = obj.evaluate(cur);
			cur[i] = cur[i] - 1.25 * step[i];
		
			smallest_error = getMinimum(error);
		
			if(smallest_error == error[0])
				step[i] = 0.73 * step[i];

			if(smallest_error == error[1])
			{
				cur[i] = cur[i] - 1.25 * step[i];
				step[i] = 1.25 * step[i];
			}
			if(smallest_error == error[2])
			{
				cur[i] = cur[i] - 0.73 * step[i];
				step[i] = 0.73 * step[i];
			}
			if(smallest_error == error[3])
			{
				cur[i] = cur[i] + 0.73 * step[i];
				step[i] = 0.73 * step[i];
			}
		
			if(smallest_error == error[4])
			{
				cur[i] = cur[i] + 1.25 * step[i];
				step[i] = 1.25 * step[i];
			}
		}
		
		return smallest_error;
	}

}
