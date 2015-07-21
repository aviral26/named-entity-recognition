import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class NER {
    
	static int ans = 0;
	
    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("Incorrect command line parameters.");
	    return;
	}	    

	String print = "";
	if (args.length > 2 && args[2].equals("-print")) {
	    print = "-print";
	}

	FeatureFactory ff = new FeatureFactory();
	List<Datum> trainData = ff.readData(args[0]);
	List<Datum> testData = ff.readData(args[1]);	
	
	// read the train and test data
	ff.readWordVectors("../data/wordVectors.txt","../data/vocab.txt");
	WindowModel model = new WindowModel(ff,6,150,0.0005);
	model.train(trainData);
	model.test(testData);
	do{
		System.out.println("\nEnter a test string: ");
		model.demonstrate(new Scanner(System.in).nextLine());
		System.out.println("\nTry again (Enter 1 to try again, else 0)? ");
		ans = new Scanner(System.in).nextInt();
	}while(ans == 1);
}
}