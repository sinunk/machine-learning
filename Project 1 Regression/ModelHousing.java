public class ModelHousing extends Function
{

    //housing
	double evaluate(double[] vals)
    {
    	//if(vals.length != 27)
            //System.out.println("housing vals.length="+vals.length);
            //throw new RuntimeException("housing: unexpected number of values");
            
        double m0 = vals[0];
        double m1 = vals[1];
        double m2 = vals[2];
        double m3 = vals[3];
        double m4 = vals[4];
        double m5 = vals[5];
        double m6 = vals[6];
        double m7 = vals[7];
        double m8 = vals[8];
        double m9 = vals[9];
        double m10 = vals[10];
        double m11 = vals[11];
        double m12 = vals[12];
        
        double b = vals[13];
        
        double x0 = vals[14];
        double x1 = vals[15];
        double x2 = vals[16];
        double x3 = vals[17];
        double x4 = vals[18];
        double x5 = vals[19];
        double x6 = vals[20];
        double x7 = vals[21];
        double x8 = vals[22];
        double x9 = vals[23];
        double x10 = vals[24];
        double x11 = vals[25];
        double x12 = vals[26];
        
	return m0*x0 + m1*x1 + m2*x2 + m3*x3 + m4*x4 + m5*x5 + m6*x6 + m7*x7 + m8*x8 + m9*x9 + m10*x10 + m11*x11 + m12*x12 + b;
    }

}