public class ModelParabola extends Function
{

	//parabola
	double evaluate(double[] vals)
    {
    	//if(vals.length != 3)
    		//System.out.println("parabola vals.length="+vals.length);
            //throw new RuntimeException("parabola: unexpected number of values");
            
        double m = vals[0];
        double b = vals[1];
        double x = vals[2];
        return m * (x * x) + b;
    }
    
}