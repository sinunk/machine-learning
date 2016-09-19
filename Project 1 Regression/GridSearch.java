public class GridSearch extends Optimizer
{
/*
Its constructor should require 3 vectors as parameters, named min, max, and step. 
Add a member variable to store each of these parameters. 
Also, add a member variable of the vector named cur, and initialize it to be a copy of min. 
Implements the iterate method like this:
Make sure you understand how these two optimizers work, or ask questions about them. 
*/
	double[] cur;
	double[] min;
	double[] max;
	double[] step;
	double[] bestYet;
	
	GridSearch(double[] mins, double[] maxes, double[] steps, Objective obj)
	{
		super(obj);
		Vec.copy(cur, mins);
		min = mins;
		max = maxes;
		step = steps;
	}
	
	double iterate()
	{
		double error = 0.0;
		error = obj.evaluate(cur);
		Vec.copy(bestYet, cur);
		
		for (int i = 0; i < cur.length - 1; i++)
		{
			cur[i] += step[i];
			if (step[i] > max[i])
				step[i] = min[i];
			else
				break;
		}
		
	return error;
	}
}