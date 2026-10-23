package com.logiciel.engine;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import com.logiciel.processing.BiblioTR;
import com.logiciel.processing.ProcLib;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class EngineImage {
	public static int size = 64; 
	
	public static String jsonPath = "ModeleONNX/classes.json";
	public static String txtPath = "ModeleONNX/Fstc_classes.txt";
	public static String modelPathDL4J = "ModeleDL4Java/model.zip";
	public static String modelPathONNX = "ModeleONNX/cnn_panneaux_frnace.onnx";
	
	
	public EngineImage() {
		
	}
	
	public static void executer(String path, String modelChoice) throws Exception{
		switch (modelChoice.toUpperCase()) {
		case "ONNX":
			executerONNXProc(path);
			break;
		case "DL4JAVA":
			executerDL4JavaProc(path);
			break;
		default:
			System.out.println("Error image: model : " + modelChoice);
		}
	}

	public static void executerONNX(String imagePath) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("Start EngineImage " + "ONNX");
		Mat image= ProcLib.LectureImage(imagePath);
		System.out.println("Lecture Image");
		//
		
		//String datasetPath = "D:\\Dataset";
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		//String[] folderCodes = BiblioTR.folderCodes(datasetPath);
		System.out.println("Chargement de classes");
		try (OrtEnvironment env = OrtEnvironment.getEnvironment();
				OrtSession session = env.createSession(modelPathONNX, new OrtSession.SessionOptions())) {
				// --- PRÉ-TRAITEMENT ---
				// Redimensionner pour le modèle RNC (64, 64) via la classe Imgproc
				Mat resized = new Mat();
				Imgproc.resize(image, resized, new Size(size, size));
				
				// BGR --> RGB pour le modèle RNC
				float[] floatData = new float[1 * size * size * 3];
				floatData = BiblioTR.bgr2rgb(floatData, resized);
				
				// --- Affichage de la prédiction RNC ---
				long[] shape = {1, size, size, 3};
		        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatData), shape)) {
		            try (OrtSession.Result results = session.run(Collections.singletonMap("input", inputTensor))) {
		                float[][] output = (float[][]) results.get(0).getValue();
		                
		                // meilleure class + proba
		                int bestClass = BiblioTR.bestClass(output);
		                float maxProb = BiblioTR.bestClassProba( output);
		                String finalName = BiblioTR.bestClassName(bestClass, indexToCode, codeToName);

		                System.out.println("Détection : " + finalName);

		                // affichage
		                
		                String text = finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
		                Core.putText(image, text, new Point(30, 50), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0, 0), 2, Core.LINE_AA, false);
		                ProcLib.afficheImage("Image", image);
		            }
		        }

		}
	}
	//
	public static void executerONNXProc(String imagePath) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("Start EngineImage " + "ONNX");
		Mat image= ProcLib.LectureImage(imagePath);
		System.out.println("Lecture Image");
		//
		//String datasetPath = "D:\\Dataset";
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		//String[] folderCodes = BiblioTR.folderCodes(datasetPath);
		//
		List<java.util.List<Mat>> toutesLesFormes = ProcLib.extractAll(image);
		if (toutesLesFormes == null || toutesLesFormes.isEmpty()) {
			System.out.println("Aucune forme détectée par extractAll.");
			ProcLib.afficheImage("Image d'origine (Aucun panneau)", image);
			image.release();
			return;
		}
		
		Mat resized = new Mat();
		int textYOffset = 50;

		try (OrtEnvironment env = OrtEnvironment.getEnvironment();
			 OrtSession session = env.createSession(modelPathONNX, new OrtSession.SessionOptions())) {
			
			for (int typeForme = 0; typeForme < toutesLesFormes.size(); typeForme++) {
				java.util.List<Mat> vignettes = toutesLesFormes.get(typeForme);
				if (vignettes == null || vignettes.isEmpty()) continue;
				
				
				String nomForme = "Inconnu";
				Scalar couleurTexte = new Scalar(0, 255, 0); 
				
				switch (typeForme) {
					case 0:
						nomForme = "Cercle";
						couleurTexte = new Scalar(0, 0, 255);
						break;
					case 1:
						nomForme = "Triangle";
						couleurTexte = new Scalar(255, 0, 0); 
						break;
					case 2:
						nomForme = "Autre";
						couleurTexte = new Scalar(0, 255, 255); 
						break;
				}
				
				
				for (int i = 0; i < vignettes.size(); i++) {
					Mat panneauMat = vignettes.get(i);
					if (panneauMat == null || panneauMat.empty()) continue;
					
					
					Imgproc.resize(panneauMat, resized, new Size(size, size));
					
					float[] floatData = new float[1 * size * size * 3];
					floatData = BiblioTR.bgr2rgb(floatData, resized);
					
					long[] shape = {1, size, size, 3};
					try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatData), shape)) {
						try (OrtSession.Result results = session.run(Collections.singletonMap("input", inputTensor))) {
							float[][] output = (float[][]) results.get(0).getValue();
							
							int bestClass = BiblioTR.bestClass(output);
							float maxProb = BiblioTR.bestClassProba(output);
							String finalName = BiblioTR.bestClassName(bestClass, indexToCode, codeToName);
							
							System.out.println("[" + nomForme + "] Panneau détecté : " + finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")");
							
							String text = nomForme + " -> " + finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
							Core.putText(image, text, new Point(30, textYOffset), Core.FONT_HERSHEY_SIMPLEX, 0.7, couleurTexte, 2, Core.LINE_AA, false);
							textYOffset += 35; 
						}
					}
					
					panneauMat.release();
				}
			}
			ProcLib.afficheImage("Résultat ONNX + extractAll", image);
			
		}finally {
			if (image != null) image.release();
			if (resized != null) resized.release();
			
			for (java.util.List<Mat> sousListe : toutesLesFormes) {
				if (sousListe != null) {
					for (Mat m : sousListe) {
						if (m != null && !m.empty()) m.release();
					}
				}
			}
		}
	}
	//
	public static void executerDL4Java(String imagePath) throws Exception{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("Start EngineImage " + "DL4Java");
		Mat image= ProcLib.LectureImage(imagePath);
		System.out.println("Lecture Image");
		//model
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPathDL4J);
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		//sizing
		Mat resized = new Mat();
		Imgproc.resize(image, resized, new Size(size, size));
		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		BufferedImage bufferedImage = ProcLib.Mat2bufferedImage(resized);
		//
		INDArray input = BiblioTR.bufferedImageToINDArray(bufferedImage);
		scaler.transform(input);
		//
		INDArray output = model.output(input);
		int classIndex = Nd4j.argMax(output, 1).getInt(0);
		double maxProb = output.getDouble(0,classIndex);
		//
		String code = indexToCode.get(String.valueOf(classIndex));
		String finalName = codeToName.get(code);
		//
		String text = finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
		Core.putText(image, text, new Point(30, 50), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0, 0), 2, Core.LINE_AA, false);
		ProcLib.afficheImage("Résultat DL4Java : " + finalName, image);
	}
	public static void executerDL4JavaProc(String imagePath) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("Start EngineImage DL4Java avec extractAll (Formes multiples)");
		
		Mat image = ProcLib.LectureImage(imagePath);
		if (image == null || image.empty()) {
			System.out.println("Erreur : Impossible de lire l'image : " + imagePath);
			return;
		}
		System.out.println("Lecture Image réussie.");
		
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPathDL4J);
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		
		java.util.List<java.util.List<Mat>> toutesLesFormes = ProcLib.extractAll(image);
		
		if (toutesLesFormes == null || toutesLesFormes.isEmpty()) {
			System.out.println("Aucune forme détectée par extractAll.");
			ProcLib.afficheImage("Image d'origine (Aucun panneau)", image);
			image.release();
			return;
		}
		
		Mat resized = new Mat();
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		int textYOffset = 50;

		try {
			for (int typeForme = 0; typeForme < toutesLesFormes.size(); typeForme++) {
				java.util.List<Mat> vignettes = toutesLesFormes.get(typeForme);
				if (vignettes == null || vignettes.isEmpty()) continue;
				
				// Identification visuelle de la forme
				String nomForme = "Inconnu";
				Scalar couleurTexte = new Scalar(0, 255, 0); // Vert par défaut
				
				switch (typeForme) {
					case 0:
						nomForme = "Cercle";
						couleurTexte = new Scalar(0, 0, 255); // Rouge
						break;
					case 1:
						nomForme = "Triangle";
						couleurTexte = new Scalar(255, 0, 0); // Bleu
						break;
					case 2:
						nomForme = "Autre";
						couleurTexte = new Scalar(0, 255, 255); // Jaune
						break;
				}
				for (int i = 0; i < vignettes.size(); i++) {
					Mat panneauMat = vignettes.get(i);
					if (panneauMat == null || panneauMat.empty()) continue;
					Imgproc.resize(panneauMat, resized, new Size(size, size));
					BufferedImage bufferedImage = ProcLib.Mat2bufferedImage(resized);
					INDArray input = BiblioTR.bufferedImageToINDArray(bufferedImage);
					scaler.transform(input);
					INDArray output = model.output(input);
					int classIndex = Nd4j.argMax(output, 1).getInt(0);
					double maxProb = output.getDouble(0, classIndex);
					// Récupération des labels
					String code = indexToCode.get(String.valueOf(classIndex));
					String finalName = codeToName.get(code);
					System.out.println("[DL4J - " + nomForme + "] Panneau détecté : " + finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")");
					// Incrustation des résultats sur l'image globale
					String text = nomForme + " -> " + finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
					Core.putText(image, text, new Point(30, textYOffset), Core.FONT_HERSHEY_SIMPLEX, 0.7, couleurTexte, 2, Core.LINE_AA, false);
					textYOffset += 35; // On descend pour la ligne suivante
					// Libération immédiate de la mémoire de la vignette
					panneauMat.release();
				}
			}
			// 3. AFFICHAGE FINAL DE L'IMAGE ANNOTÉE
			ProcLib.afficheImage("Résultat DL4Java + extractAll", image);
		} finally {
			// Nettoyage impératif des ressources OpenCV pour éviter les fuites de mémoire
			if (image != null) image.release();
			if (resized != null) resized.release();
			
			// Nettoyage de sécurité en boucle si un crash est survenu au milieu du traitement
			for (java.util.List<Mat> sousListe : toutesLesFormes) {
				if (sousListe != null) {
					for (Mat m : sousListe) {
						if (m != null && !m.empty()) m.release();
					}
				}
			}
		}
	}
}
