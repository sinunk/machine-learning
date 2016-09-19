public class HillClimber extends Optimizer 
{
/*
This class should have two member variables, named something like cur and step. 
Both of these should be vectors of doubles. 
The constructor should let the user specify the size of these vectors, 
and should initilize each element of cur to 0.0, and each element of step to 0.1. 
Implement the iterate method according to the following pseudocode:
(Note that the values 1.25 and 0.8 are not necessarily ideal. 
You can make these parameters, if you like.)
*/

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

		//debugging
		//cur[0] = 0.5;
		//cur[1] = 2.0;
		
		step = new double[stepSize];			
		for(int i = 0; i < stepSize; i++)
			step[i] = 0.1;
	
	}

	double iterate()
	{
		if (cur==null)
		throw new RuntimeException("cur should not be null");
		//System.out.print("Before: ");
        //Vec.println(cur);
		
		
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
		
		//System.out.print("After: ");
        //Vec.println(cur);
		return smallest_error;
	}

}