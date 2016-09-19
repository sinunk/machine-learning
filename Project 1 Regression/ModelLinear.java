public class ModelLinear extends Function
{	
	double evaluate(double[] vals)
    {	
    	//if(vals.length != 3)
    		//System.out.println("linear vals.length="+vals.length);
            //throw new RuntimeException("linear: unexpected number of values");
            
        double m = vals[0];
        double b = vals[1];
        double x = vals[2];
        //System.out.println("m="+m+",b="+b);
        return m * x + b;
    }

}