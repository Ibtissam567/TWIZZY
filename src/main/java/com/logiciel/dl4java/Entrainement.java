package com.logiciel.dl4java;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;


import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.linalg.learning.config.Adam;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.deeplearning4j.*;


import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;

import com.logiciel.processing.BiblioTR;

public class Entrainement {


	public static void main(String[] args) throws Exception {
		String jsonPath = "D:\\AjoutAli\\classes.json";
		String txtPath = "D:\\AjoutAli\\Fstc_classes.txt";
		String datasetPath = "D:\\AjoutAli\\classes_tree";
		
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		String[] folderCodes = BiblioTR.folderCodes(datasetPath);
		
		
		// --- CONFIGURATION --------------------
        int height = 64;       
        int width = 64;
        int channels = 3;      
        int batchSize = 32;    
        

        int numClasses = folderCodes.length;   
        int rngSeed = 42;
        Random randNumGen = new Random(rngSeed);
        
		// -------------------------------------
		
        File parentDir = new File(datasetPath);
        FileSplit filesInDir = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        
        // Scan des images
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, new  ParentPathLabelGenerator());
        recordReader.initialize(filesInDir);

        //--------------- Split -------------------------------------------
        /*
        splitDataset(
        		datasetPath,
        	    "D:/AjoutAli/train",
        	    "D:/AjoutAli/test",
        	    0.8
        	);
        */
        FileSplit trainSplit = new FileSplit(new File("D:/AjoutAli/train"),NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        
        FileSplit testSplit = new FileSplit(new File("D:/AjoutAli/test"),NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        
        
        List<String> labelsOfficiels = recordReader.getLabels();
        System.out.println(labelsOfficiels.get(0));
        
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

	     // Split équilibré
        
        
        BalancedPathFilter pathFilter =new BalancedPathFilter(randNumGen, labelMaker, 0);

	    InputSplit[] filesSplit = filesInDir.sample(pathFilter, 80, 20);

	    // Nombre total de classes
	    ImageRecordReader tmpReader = new ImageRecordReader(height, width, channels, labelMaker);
	    tmpReader.initialize(filesInDir);
	
	    int numClasse = tmpReader.getLabels().size();
	
	    System.out.println("Nombre classes : " + numClasse);
	
	    // TRAIN
	    ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
	    trainRecordReader.initialize(trainSplit);
	    DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainRecordReader,batchSize,1,numClasses);
	
	    // TEST
	    ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
	    testRecordReader.initialize(testSplit);
	    DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader,batchSize,1,numClasses);
        // ----------------------------------------------------------------------
	    
	    System.out.println(trainRecordReader.getLabels().size());
	    System.out.println(testRecordReader.getLabels().size());
        
        
        // Normalisation : Ramener les pixels de [0, 255] à [0, 1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        // --- 2. ARCHITECTURE CNN (Adaptée pour panneaux routiers) ---
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new org.nd4j.linalg.learning.config.Adam.Builder()
                        .learningRate(0.001)
                        .beta1(0.9)
                        .beta2(0.999)
                        .epsilon(1e-8)
                        .build())
                .list()
                // Couche 1 : Convolution
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                // Couche 2 : Max Pooling (Réduction de taille de moitié)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Couche 3 : Deuxième Convolution
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                // Couche 4 : Deuxième Max Pooling
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Couche 5 Couche dense 
                .layer(4, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                
                // Couche 6 Sortie
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        // --- 3. ENTRAÎNEMENT ---
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        
        //UIServer uiServer = UIServer.getInstance();
        //StatsStorage statsStorage = new InMemoryStatsStorage();

        //uiServer.attach(statsStorage);
        model.setListeners(
            //new StatsListener(statsStorage),
            new ScoreIterationListener(10)
        );
        
        model.init();
        
        System.out.println("Début de l'entraînement...");
        int epochs = 5; 
        for (int i = 0; i < epochs; i++) {
            System.out.println("Époque " + (i + 1) + " / " + epochs);
            model.fit(trainIterator);
            trainIterator.reset(); 
        }
        
        File modelFile = new File("D:/AjoutAli/model.zip");
        ModelSerializer.writeModel(model, modelFile, true);

        System.out.println("Sauvegarde");

        // --- 4. ÉVALUATION ---
        System.out.println("Évaluation du modèle...");
        Evaluation eval = model.evaluate(testIterator);
        System.out.println(eval.stats());
    }
	
	public static void splitDataset(
	        String sourceDir,
	        String trainDir,
	        String testDir,
	        double trainRatio
	) throws IOException {

	    Random random = new Random(42);

	    File source = new File(sourceDir);

	    for (File classDir : source.listFiles(File::isDirectory)) {

	        File[] images = classDir.listFiles();

	        if (images == null || images.length == 0)
	            continue;

	        List<File> imageList = Arrays.asList(images);

	        Collections.shuffle(imageList, random);

	        int trainSize = Math.max(1,
	                (int)(imageList.size() * trainRatio));

	        File trainClassDir =
	                new File(trainDir, classDir.getName());

	        File testClassDir =
	                new File(testDir, classDir.getName());

	        trainClassDir.mkdirs();
	        testClassDir.mkdirs();

	        for (int i = 0; i < imageList.size(); i++) {

	            File src = imageList.get(i);

	            File dest;

	            if (i < trainSize) {
	                dest = new File(trainClassDir, src.getName());
	            } else {
	                dest = new File(testClassDir, src.getName());
	            }

	            Files.copy(
	                    src.toPath(),
	                    dest.toPath(),
	                    StandardCopyOption.REPLACE_EXISTING
	            );
	        }
	    }
	}
}
