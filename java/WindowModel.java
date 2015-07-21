import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class WindowModel {

	protected SimpleMatrix Wv, W, Wout;
	public int windowSize,wordSize, hiddenSize;


	public HashMap<String, Integer> wordToNum;
	public HashMap<Integer, String> numToWord;
	List<Datum> trainData;

	// learning rate
	double lr;

	protected int maxIter, miniBatchSize;
	protected double funcTol, regC, regC_Wv, regC_Wout;
	protected boolean trainAllParams, doGradientCheck;
	protected int optimizationMethod;

	protected int currentOptimizerIter;
	protected Random rgen = new Random();
	
	BufferedWriter to;
	BufferedWriter so;
	
	public WindowModel(FeatureFactory ff,int _windowSize, int _hiddenSize, double _lr){
		Wv = ff.allVecs;
		wordToNum = ff.wordToNum; 
		numToWord = ff.numToWord;
		wordSize = Wv.numRows();

		//learning rate
		lr = _lr;
		windowSize = _windowSize;
		hiddenSize = _hiddenSize;
		initWeights();
	}

	public void initWeights(){

		int fanIn = wordSize*windowSize;
		// initialize with bias inside as the last column
		W = SimpleMatrix.random(hiddenSize,fanIn+1, -1/Math.sqrt(fanIn), 1/Math.sqrt(fanIn), rgen);

		//random vector
		Wout = SimpleMatrix.random(1,hiddenSize, -1/Math.sqrt(fanIn), 1/Math.sqrt(fanIn), rgen);
	}


	public void train(List<Datum> _trainData ) throws IOException{
		trainData = _trainData;
		int totalIter = 1;
		
		try{
			to = new BufferedWriter(new FileWriter("E:\\7th semester project\\trainOutput.txt"));
			so = new BufferedWriter(new FileWriter("E:\\7th semester project\\scoreOutput.txt"));
		}catch(Exception e){
			System.out.println("Cannot output to file.");
		}
		
		int numWordsInTrain = trainData.size();
		System.out.println("\nExecuting..");
		for (int iter=0;iter<totalIter;iter++){

			for (int i = 0;i<numWordsInTrain;i++){

				Datum datum = trainData.get(i);
				int y;
				if (datum.label.equals("O")){
					y = 0;
				} 
				else {
					y= 1;
				}

				// forward propagation
				int[] windowNums = getWindowNumsTrain(i); //windowNums contains the vocab indexes of the words in the current window
				SimpleMatrix allX = getWindowVectorWithBias(windowNums); // input vector X with bias- a_1
				SimpleMatrix h = tanh(W.mult(allX)); //activation a- a_2
				double p_pred = sigmoid((double)Wout.mult(h).get(0)); //computed prediction
				//forward propagation complete
				// compute derivatives
				SimpleMatrix Wout_df = h.scale(y-p_pred); //partial derivative wrt Wout parameter
				SimpleMatrix allXT = allX.transpose(); //take transpose of activations vector a
				
				//SimpleMatrix delta = Wout.mult(tanhDer(h).scale(y-p_pred));//delta term for hidden layer
				SimpleMatrix temp = tanhDer(h).scale(y-p_pred);
				SimpleMatrix delta = new SimpleMatrix(hiddenSize,1);
				for(int d=0;d<hiddenSize;d++){
					delta.set(d,0,(Wout.get(0,d)*temp.get(d,0)));
				}
				
				SimpleMatrix W_df =  delta.mult(allXT); //partial derivative wrt W parameter.
				
				SimpleMatrix wordVector_df = delta.transpose().mult(W);
				
				if(!gradientCheck(Wout_df, Wout, W_df, W)){
					System.out.println("#"+i+"Gradient check failed");
					return;
				}
				
				
				// update with simple SGD step
				Wout = Wout.plus(Wout_df.scale(lr).transpose());
				//Wout = Wout.minus(Wout_df.scale(lr).transpose());
				W = W.plus(W_df.scale(lr));
				//W = W.minus(W_df.scale(lr));
				
				//SimpleMatrix updatedX = allX.minus(wordVector_df.scale(lr).transpose()); // updated word vectors
				SimpleMatrix updatedX = new SimpleMatrix(allX.numRows(),allX.numCols());
				SimpleMatrix temp2 = wordVector_df.scale(lr);
				for(int k=0;k<temp2.numRows();k++){
					for(int l=0;l<temp2.numCols();l++){
						updatedX.set(k+l,0,allX.get(k+l,0)-temp2.get(k,l));
					}
				}
				
				updateWordVectors(updatedX,windowNums);
				
				// check if prob is higher?
				allX = getWindowVectorWithBias(windowNums);
				h = tanh(W.mult(allX));
				double p_predNew = sigmoid((double)Wout.mult(h).get(0));
				//System.out.println("#"+i+ " Label:"+Integer.toString(y)+" Old: "+Double.toString(p_pred)+", new: "+Double.toString(p_predNew));				
				//System.out.println(".");
				to.write("#" + i + " Label:"+Integer.toString(y)+" Old: "+Double.toString(p_pred)+", new: "+Double.toString(p_predNew));
			}
			
			//test on train set
			int tp=0;
			int tn=0;
			int fp=0;
			int fn=0;
			for (int i = 0;i<numWordsInTrain;i++){
				
				Datum datum = trainData.get(i);
				int y;
				if (datum.label.equals("O")){y = 0;} else {y= 1;}
				// forward prop
				int[] windowNums = getWindowNumsTrain(i);
				SimpleMatrix allX = getWindowVectorWithBias(windowNums);
				SimpleMatrix h = tanh(W.mult(allX));
				double p_pred = sigmoid((double)Wout.mult(h).get(0));
				if (p_pred>0.5 && y==1){
					tp++;
				}else if (p_pred>0.5 && y==0) {
					fp++;
				}else if (p_pred<=0.5 && y==0) {
					tn++;
				}else if (p_pred<=0.5 && y==1) {
					fn++;
				}
			}
			double prec = (double)tp/(tp+fp);
			double rec = (double)tp/(tp+fn);
			double f1 = (double)2.0*prec*rec/(prec+rec);
			System.out.println("Training Precision="+Double.toString(prec)+", Recall="+Double.toString(rec)+", F1="+Double.toString(f1));
			so.write("Train: Training Precision="+Double.toString(prec)+", Recall="+Double.toString(rec)+", F1="+Double.toString(f1));
		}
		System.out.println("\nFinished training.");
	}

	private boolean gradientCheck(SimpleMatrix Wout_df, SimpleMatrix Wout, SimpleMatrix W_df, SimpleMatrix W){
		
		
		
		return true;
	}
	
	private void updateWordVectors(SimpleMatrix updatedX, int[] windowNums){
		int k=0;
		for(int j : windowNums){
			for(int i=0;i<50;i++){
				Wv.set(i,j,updatedX.get(k,0));
				k++;
			}
		}
	}
	
	public void test(List<Datum> testData) throws IOException{
	int tp=0;
	int tn=0;
	int fp=0;
	int fn=0;
	int numWordsInTrain = testData.size();

	System.out.println("Testing..");
	for (int i = 0;i<numWordsInTrain;i++){
		Datum datum = testData.get(i);
		int y;
		if (datum.label.equals("O")){y = 0;} else {y= 1;}
		// forward prop
		int[] windowNums = getWindowNumsTest(i,testData);
		SimpleMatrix allX = getWindowVectorWithBias(windowNums);
		SimpleMatrix h = tanh(W.mult(allX));
		double p_pred = sigmoid((double)Wout.mult(h).get(0));
		if (p_pred>0.5 && y==1){
			tp++;
		}else if (p_pred>0.5 && y==0) {
			fp++;
		}else if (p_pred<=0.5 && y==0) {
			tn++;
		}else if (p_pred<=0.5 && y==1) {
			fn++;
		}
	}
	double prec = (double)tp/(tp+fp);
	double rec = (double)tp/(tp+fn);
	double f1 = (double)2.0*prec*rec/(prec+rec);
	//System.out.println("Test: Precision="+Double.toString(prec)+", Recall="+Double.toString(rec)+", F1="+Double.toString(f1));
	so.write("\nTest: Precision="+Double.toString(prec)+", Recall="+Double.toString(rec)+", F1="+Double.toString(f1));
	System.out.println("Finished testing.");
	}
	
	public void demonstrate(String testString){
		
		ArrayList<String> testData = new ArrayList<String>();
		String[] temp = testString.split(" ");
		for(String x : temp){
			testData.add(x);
		}
		for(int i=0;i<testData.size();i++){
			int[] windowNums = getWindowNumsDemonstrate(i,testData);
			SimpleMatrix allX = getWindowVectorWithBias(windowNums);
			SimpleMatrix h = tanh(W.mult(allX));
			double p_pred = sigmoid((double)Wout.mult(h).get(0));
			System.out.print("\n" + testData.get(i)  + " ");
			if(p_pred >= 0.5){
				System.out.print("NAMED ENTITY");
			}
			else{
				System.out.println("NOT A NAMED ENTITY");
			}
			System.out.println(" Predicted value: " + p_pred);
		}
	}
	
	private int[] getWindowNumsDemonstrate(int wordPos,ArrayList<String> testData) {
		int[] windowNums = new int[windowSize];
		int startSymbol = wordToNum.get("<s>");
		int endSymbol = wordToNum.get("</s>");
		int contextSize = (int) Math.floor((windowSize-1)/2);
		int counter = 0;
		
		for (int i=wordPos-contextSize;i<=wordPos+contextSize;i++){
			if (i<0){
				windowNums[counter] =startSymbol;
			} else if (i>=testData.size()){
				windowNums[counter] =endSymbol;
			} else {
				windowNums[counter] = getWordIDDemonstrate(i,testData);
			}
			counter++;
		}

		return windowNums;
	}
	
	public int getWordIDDemonstrate(int position,ArrayList<String> testData){
		int out;
		
		try{
			//System.out.println("Current word: " + testData.get(position));
			out = wordToNum.get(testData.get(position));
			//System.out.println("Setting out to " + out);
		} catch (Exception e){
			//System.out.println("Position: " + position);
			System.out.println("\nGiven word \"" + testData.get(position) +"\" not found in dictionary. Erroneous result may be produced.");
			out = 0;
			
		}
		return out;
	}
	
	private SimpleMatrix getWindowVectorWithBias(int[] windowNums) {//returns input vector X
		SimpleMatrix allX= new SimpleMatrix(wordSize*windowSize+1,1); 
		for (int i = 0;i<windowSize;i++){
			allX.insertIntoThis(i*wordSize, 0, Wv.extractVector(false, windowNums[i]));			
		}
		// adding bias
		allX.set(allX.numRows()-1, 0, 1);
		return allX;
	}

	private int[] getWindowNumsTest(int wordPos,List<Datum> testData) {
		int[] windowNums = new int[windowSize];
		int startSymbol = wordToNum.get("<s>");
		int endSymbol = wordToNum.get("</s>");
		int contextSize = (int) Math.floor((windowSize-1)/2);
		int counter = 0;
		for (int i=wordPos-contextSize;i<=wordPos+contextSize;i++){
			if (i<0){
				windowNums[counter] =startSymbol;
			} else if (i>testData.size()){
				windowNums[counter] =endSymbol;
			} else {
				windowNums[counter] = getWordIDTest(i,testData);
			}
			counter++;
		}

		return windowNums;
	}
	
	
	
	private int[] getWindowNumsTrain(int wordPos) {
		int[] windowNums = new int[windowSize];
		int startSymbol = wordToNum.get("<s>");
		int endSymbol = wordToNum.get("</s>");
		int contextSize = (int) Math.floor((windowSize-1)/2);
		int counter = 0;
		for (int i=wordPos-contextSize;i<=wordPos+contextSize;i++){
			if (i<0){
				windowNums[counter] =startSymbol;
			} else if (i>trainData.size()){
				windowNums[counter] =endSymbol;
			} else {
				windowNums[counter] = getWordIDTrain(i);
			}
			counter++;
		}

		return windowNums;
	}

	public int getWordIDTest(int position,List<Datum> testData){
		int out;
		try{
			out = wordToNum.get(testData.get(position).word);
		} catch (Exception e){
			// UNK=0
			out = 0;
		}
		return out;
	}
	
	
	
	public int getWordIDTrain(int position){
		int out;
		try{
			out = wordToNum.get(trainData.get(position).word);
		} catch (Exception e){
			// UNK=0
			out = 0;
		}
		return out;
	}

	//element wise tanh
	public SimpleMatrix tanh(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++)
				out.set(i,j,Math.tanh(in.get(i,j)));
		return out;
	}	

	//derivative function
	public SimpleMatrix tanhDer(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		out.set(1);
		out.set(out.minus(in.elementMult(in)));
		return out;
	}	

	//element wise sigmoid
	public SimpleMatrix sigmoid(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++)
				out.set(i,j,sigmoid(in.get(i,j)));
		return out;
	}	

	
	public SimpleMatrix sigmoidDer(SimpleMatrix in){
		SimpleMatrix ones = new SimpleMatrix(in.numRows(),in.numCols());
		ones.set(1);
		return in.elementMult(ones.minus(in));
	}		


	public static double sigmoid(double x) {
		return (1 / (1 + Math.exp(-x)));
	}

	//element wise tanh
	public static void elemTanh(SimpleMatrix in, SimpleMatrix out){
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++)
				out.set(i,j,Math.tanh(in.get(i,j)));
	}


}
