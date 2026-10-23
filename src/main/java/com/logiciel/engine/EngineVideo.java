package com.logiciel.engine;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import com.logiciel.processing.BiblioTR;
import com.logiciel.processing.ProcLib;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class EngineVideo {
	public static int size = 64;
	public static String jsonPath = "ModeleONNX/classes.json";
	public static String txtPath = "ModeleONNX/Fstc_classes.txt";
	public static String modelPathDL4J = "ModeleDL4Java/model.zip";
	public static String modelPathONNX = "ModeleONNX/cnn_panneaux_frnace.onnx";
	
	
	
	
	public EngineVideo() {
		// TODO Auto-generated constructor stub
	}
	//
	public static void executer(String path, String modelChoice) throws Exception{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		switch (modelChoice.toUpperCase()) {
		case "ONNX":
			executerONNXProc(path);
			break;
		case "DL4JAVA":
			executerDL4Java(path);
			break;
		default:
			System.out.println("Error image: model : " + modelChoice);
		}
	}
	private static void executerDL4Java(String videoPath) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("Start EngineVideo DL4Java");
		//
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPathDL4J);
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		System.out.println("Chargement des classes DL4Java complété.");
		//
		VideoCapture camera = new VideoCapture(videoPath);
		if (!camera.isOpened()) {
			System.out.println("Erreur vidéo : file" + videoPath);
			return;
		}
		System.out.println("video chargee (DL4Java)");
		//
		JFrame frame = new JFrame("Panneaux Recognition - Video DL4Java");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); 
		JLabel label = new JLabel();
		frame.getContentPane().add(label);
		frame.setSize(720, 480);
		frame.setLocationRelativeTo(null); 
		frame.setVisible(true);
		//
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		//
		new Thread(() -> {
			Mat image = new Mat();
			Mat resized = new Mat();
			try {
				while (frame.isVisible()) {
					// Capturer la frame
					if (!camera.read(image)) {
						System.out.println("Fin de la vidéo ou échec de la capture.");
						break;
					}
					// --- PRÉ-TRAITEMENT POUR L'IA (Miniature 64x64) ---
					Imgproc.resize(image, resized, new Size(size, size));
					//
					// Conversion en BufferedImage pour le convertisseur de ton collègue
					BufferedImage bufferedImageResized = ProcLib.Mat2bufferedImage(resized);

					// Conversion en tenseur DL4J (INDArray) 
					INDArray input = BiblioTR.bufferedImageToINDArray(bufferedImageResized);
					//
					// Application de la normalisation (0-1) sur les pixels
					if (scaler != null) {
						scaler.transform(input);
					}
					//
					// --- INFÉRENCE DL4J ---
					INDArray output = model.output(input);
					int classIndex = Nd4j.argMax(output, 1).getInt(0);
					double maxProb = output.getDouble(0, classIndex);
					//
					String code = indexToCode.get(String.valueOf(classIndex));
					String finalName = codeToName.get(code);
					//
					String text = finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
					Core.putText(image, text, new Point(30, 50), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0, 0), 2, Core.LINE_AA, false);
					// --- CONVERSION ET RENDER VIA LE HANDLER SWING ---
					BufferedImage bufferedImageFinal = ProcLib.Mat2bufferedImage(image);
					if (bufferedImageFinal != null) {
						javax.swing.SwingUtilities.invokeLater(() -> {
							label.setIcon(new ImageIcon(bufferedImageFinal));
							label.repaint();
						});
					}
					//Sleep, set fps
					Thread.sleep(34);
				}
			} catch (Exception e) {
				System.out.println("Erreur pendant la lecture DL4J : " + e.getMessage());
				e.printStackTrace();
			} finally {
				// Nettoyage complet
				System.out.println("Libération des ressources DL4Java.");
				image.release();
				camera.release();
				frame.dispose();
				resized.release();
			}
		}).start();
		//
	}
	//
	public static void executerONNX(String videoPath) throws Exception {
		System.out.println("Start EngineVideo ONNX");
		//
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		//
		System.out.println("Chargement de classes");
		//
		OrtEnvironment env = OrtEnvironment.getEnvironment();
		OrtSession session = env.createSession(modelPathONNX, new OrtSession.SessionOptions());
		//
		VideoCapture camera = new VideoCapture(videoPath);
		if (!camera.isOpened()) {
			System.out.println("Erreur : Impossible d'ouvrir la video : " + videoPath);
			return;
		}
		System.out.println("Video chargee avec succes.");
		
		JFrame frame = new JFrame("Panneaux Recognition - Video ONNX");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Ne coupe pas le menu à la fermeture
		JLabel label = new JLabel();
		frame.getContentPane().add(label);
		frame.setSize(720, 480);
		frame.setLocationRelativeTo(null); // Centre la fenêtre à l'écran
		frame.setVisible(true);
		// THREAD /!\
		new Thread(() -> {
			Mat image = new Mat();
			Mat resized = new Mat();
			try {
				float[] floatData = new float[1 * size * size * 3];
				while (frame.isVisible()) {
					// Capturer la frame de la vidéo
					if (!camera.read(image)) {
						System.out.println("Fin de la vidéo ou échec de la capture.");
						break;
					}
					
					//
					
					Imgproc.resize(image, resized, new Size(size, size));
					//
					floatData = BiblioTR.bgr2rgb(floatData, resized);
					
					// --- INFÉRENCE IA ---
					// Cette méthode écrit directement le texte sur la GRANDE matrice "image"
					BiblioTR.affichagePredictionRNC(indexToCode, codeToName, floatData, env, session, image);
					
					// --- CONVERSION DE LA GRANDE IMAGE POUR SWING ---
					BufferedImage bufferedImage = ProcLib.Mat2bufferedImage(image);
					
					if (bufferedImage != null) {
						// On demande gentiment à Swing de mettre à jour l'image dès qu'il a un moment
						javax.swing.SwingUtilities.invokeLater(() -> {
							label.setIcon(new ImageIcon(bufferedImage));
							label.repaint();
						});
					}
					// Un micro-délai (ex: 30ms) pour laisser le processeur respirer et caler la vidéo à ~30 FPS
					Thread.sleep(34);
					}
			} catch (Exception e) {
				System.out.println("Erreur pendant la lecture du flux vidéo : " + e.getMessage());
				e.printStackTrace();
			} finally {
				// Nettoyage complet à la fermeture de la fenêtre ou à la fin du film
				System.out.println("Fermeture du flux vidéo et libération des ressources.");
				image.release();
				camera.release();
				frame.dispose();
				resized.release();
				try {
					session.close();
					env.close();
				} catch (Exception e) {
					// Ignore
				}
			}
		}).start(); //threading  
	}
	//
	public static void executerONNXProc(String videoPath) throws Exception {
		System.out.println("Start EngineVideo ONNX proc");
		
		Map<String, String> indexToCode = BiblioTR.indexToCode(jsonPath);
		Map<String, String> codeToName = BiblioTR.codeToName(txtPath);
		
		System.out.println("Chargement de classes");
		
		OrtEnvironment env = OrtEnvironment.getEnvironment();
		OrtSession session = env.createSession(modelPathONNX, new OrtSession.SessionOptions());
		
		VideoCapture camera = new VideoCapture(videoPath);
		if (!camera.isOpened()) {
			System.out.println("Erreur : Impossible d'ouvrir la video : " + videoPath);
			session.close();
			env.close();
			return;
		}
		System.out.println("Video chargee avec succes.");
		
		JFrame frame = new JFrame("Panneaux Recognition - Video ONNX (Proc)");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); 
		JLabel label = new JLabel();
		frame.getContentPane().add(label);
		frame.setSize(720, 480);
		frame.setLocationRelativeTo(null); 
		frame.setVisible(true);
		
		// THREAD /!\
		new Thread(() -> {
			Mat image = new Mat();
			Mat resized = new Mat();
			Mat rgbMat = new Mat(); // Matrice intermédiaire pour la conversion RGB sécurisée
			List<List<Mat>> toutesLesFormes = null;
			
			try {
				long[] shapeVignette = {1, size, size, 3};
				
				while (frame.isVisible()) {
					// Capturer la frame de la vidéo
					if (!camera.read(image)) {
						System.out.println("Fin de la vidéo ou échec de la capture.");
						break;
					}
					
					// 1. EXTRACTALL SUR LA FRAME ACTUELLE
					toutesLesFormes = ProcLib.extractAll(image);
					int textYOffset = 50; 
					
					if (toutesLesFormes != null && !toutesLesFormes.isEmpty()) {
						
						// 2. RECHERCHE DES PANNEAUX DANS LES VIGNETTES DÉTECTÉES
						for (int typeForme = 0; typeForme < toutesLesFormes.size(); typeForme++) {
							List<Mat> vignettes = toutesLesFormes.get(typeForme);
							if (vignettes == null || vignettes.isEmpty()) continue;
							
							String nomForme = "Inconnu";
							Scalar couleurTexte = new Scalar(0, 255, 0); 
							switch (typeForme) {
								case 0: nomForme = "Cercle"; couleurTexte = new Scalar(0, 0, 255); break;
								case 1: nomForme = "Triangle"; couleurTexte = new Scalar(255, 0, 0); break;
								case 2: nomForme = "Autre"; couleurTexte = new Scalar(0, 255, 255); break;
							}
							
							for (int i = 0; i < vignettes.size(); i++) {
								Mat panneauMat = vignettes.get(i);
								if (panneauMat == null || panneauMat.empty()) continue;
								
								// A. Redimensionner la vignette au format du CNN (ex: 64x64)
								Imgproc.resize(panneauMat, resized, new Size(size, size));
								
								// B. Convertir proprement de BGR (OpenCV) vers RGB (Format IA)
								Imgproc.cvtColor(resized, rgbMat, Imgproc.COLOR_BGR2RGB);
								
								// C. Remplir un tableau de floats tout neuf (Élimine la rémanence du "STOP")
								float[] floatDataVignette = new float[size * size * 3];
								int idx = 0;
								for (int y = 0; y < size; y++) {
									for (int x = 0; x < size; x++) {
										double[] pixel = rgbMat.get(y, x);
										floatDataVignette[idx++] = (float) (pixel[0] / 255.0); // R
										floatDataVignette[idx++] = (float) (pixel[1] / 255.0); // G
										floatDataVignette[idx++] = (float) (pixel[2] / 255.0); // B
									}
								}
								
								// D. Inférence ONNX sur le tableau de la vignette
								try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatDataVignette), shapeVignette)) {
									try (OrtSession.Result results = session.run(Collections.singletonMap("input", inputTensor))) {
										float[][] output = (float[][]) results.get(0).getValue();
										
										int bestClass = BiblioTR.bestClass(output);
										float maxProb = BiblioTR.bestClassProba(output);
										
										// Seuil de confiance minimal à 45% pour éviter les fausses prédictions folles
										if (maxProb > 0.45) {
											String finalName = BiblioTR.bestClassName(bestClass, indexToCode, codeToName);
											
											// Écriture des prédictions sur la grande image de la vidéo
											String text = nomForme + " -> " + finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
											Core.putText(image, text, new Point(30, textYOffset), Core.FONT_HERSHEY_SIMPLEX, 0.6, couleurTexte, 2, Core.LINE_AA, false);
											textYOffset += 30;
										}
									}
								}
								
								// NETTOYAGE IMPÉRATIF de la vignette après usage
								panneauMat.release();
							}
						}
					}
					
					// --- CONVERSION ET AFFICHAGE DANS L'INTERFACE SWING ---
					BufferedImage bufferedImage = ProcLib.Mat2bufferedImage(image);
					if (bufferedImage != null) {
						javax.swing.SwingUtilities.invokeLater(() -> {
							label.setIcon(new ImageIcon(bufferedImage));
							label.repaint();
						});
					}
					
					// Libération de la liste de sous-images pour cette frame
					if (toutesLesFormes != null) {
						for (List<Mat> sousListe : toutesLesFormes) {
							if (sousListe != null) {
								for (Mat m : sousListe) {
									if (m != null && !m.empty()) m.release();
								}
							}
						}
					}
					
					// Un micro-délai pour caler la vidéo à ~30 FPS
					Thread.sleep(34);
				}
			} catch (Exception e) {
				System.out.println("Erreur pendant la lecture du flux vidéo : " + e.getMessage());
				e.printStackTrace();
			} finally {
				System.out.println("Fermeture du flux vidéo et libération des ressources.");
				if (image != null) image.release();
				if (resized != null) resized.release();
				if (rgbMat != null) rgbMat.release();
				
				if (toutesLesFormes != null) {
					for (List<Mat> sousListe : toutesLesFormes) {
						if (sousListe != null) {
							for (Mat m : sousListe) {
								if (m != null && !m.empty()) m.release();
							}
						}
					}
				}
				
				camera.release();
				frame.dispose();
				try {
					session.close();
					env.close();
				} catch (Exception e) {
					// Ignore
				}
			}
		}).start(); // threading  
	}
}