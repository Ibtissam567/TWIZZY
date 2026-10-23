package com.logiciel.dl4java;

import java.awt.image.BufferedImage;
import java.util.Map;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import com.logiciel.processing.BiblioTR;
import com.logiciel.processing.ProcLib;

public class ReelTimePred {
	
	public static void execution(String[] args) throws Exception{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("ModeleDL4Java/model.zip");
		String jsonPath = "ModeleONNX/classes.json";
		String txtPath = "ModeleONNX/Fstc_classes.txt";
		//String datasetPath = "D:\\Dataset";
		
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		//String[] folderCodes = BiblioTR.folderCodes(datasetPath);
		
		System.out.println("Chargement de classes");
		VideoCapture camera = new VideoCapture(0);
		if (!camera.isOpened()) {
			System.out.println("Erreur : Impossible d'ouvrir la webcam.");
			return;
		}
		System.out.println("Webcam démarrée.");

		// Création d'une fenêtre d'affichage Java Swing standard
		JFrame frame = new JFrame("Panneaux Recognition - Real Time");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JLabel label = new JLabel();
		frame.getContentPane().add(label);
		frame.setSize(640, 480);
		frame.setVisible(true);
        

		DataNormalization scaler = new ImagePreProcessingScaler(0,1);

		
		Mat image = new Mat();
		while (frame.isVisible()) {
			// Capturer l'image de la webcam
			if (!camera.read(image)) {
				System.out.println("Échec de la capture d'image.");
				break;
			}
            
			// --- PRÉ-TRAITEMENT ---
			// Redimensionner pour le modèle RNC (64, 64) via la classe Imgproc
			Mat resized = new Mat();
			Imgproc.resize(image, resized, new Size(64, 64));

			BufferedImage bufferedImage = ProcLib.Mat2bufferedImage(resized);

			INDArray input =bufferedImageToINDArray(bufferedImage);
			scaler.transform(input);
            
			// --- prédiction RNC ---
			INDArray output = model.output(input);
			int classIndex = Nd4j.argMax(output, 1).getInt(0);
			double maxProb = output.getDouble(0,classIndex);
			
			String code = indexToCode.get(String.valueOf(classIndex));
			String finalName = codeToName.get(code);
			
			System.out.println("OUTPUT: " + classIndex);
			
			// Affichage
			String text = finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
            Core.putText(image, text, new Point(30, 50), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0, 0), 2, Core.LINE_AA, false);
            
            if (bufferedImage != null) {
				label.setIcon(new ImageIcon(bufferedImage));
				frame.repaint();
			}
		}
        
		camera.release(); // Libère la webcam
		frame.dispose();
		
	}
	
	public static INDArray bufferedImageToINDArray(BufferedImage image) {

	    int width = image.getWidth();
	    int height = image.getHeight();

	    INDArray array = Nd4j.create(1, 3, height, width);

	    for (int y = 0; y < height; y++) {
	        for (int x = 0; x < width; x++) {

	            int rgb = image.getRGB(x, y);

	            int r = (rgb >> 16) & 0xFF;
	            int g = (rgb >> 8) & 0xFF;
	            int b = rgb & 0xFF;

	            array.putScalar(new int[]{0, 0, y, x}, r / 255.0);
	            array.putScalar(new int[]{0, 1, y, x}, g / 255.0);
	            array.putScalar(new int[]{0, 2, y, x}, b / 255.0);
	        }
	    }

	    return array;
	}

}
