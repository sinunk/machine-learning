public class GridSearch extends Optimizer
{

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
