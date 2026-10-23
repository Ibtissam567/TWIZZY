package com.logiciel.processing;

import ai.onnxruntime.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import static org.opencv.imgproc.Imgproc.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


import com.fasterxml.jackson.databind.ObjectMapper;


public class BiblioTR {
	
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
	
	public static Map<String, String> indexToCode(String jsonPath) throws Exception {
		/// fonction qui fait la correspondance index --> classes codés
		ObjectMapper mapper = new ObjectMapper();
        Map<String, String> indexToCode = mapper.readValue(new File(jsonPath), Map.class);
		
        return indexToCode;
	}
	
	public static Map<String, String> codeToName(String txtPath) throws Exception {
		/// fonction qui fait la correspondance classes codés --> nom lisible (ex : A1a -> Virage dangereux a droite)

	    Map<String, String> map = new HashMap<>();
	    
	    // Expression régulière pour capturer "Code": "Description"
	    Pattern pattern = Pattern.compile("\"(.+?)\":\\s*\"(.+?)\"");

	    Files.lines(Paths.get(txtPath)).forEach(line -> {
	        Matcher matcher = pattern.matcher(line);
	        if (matcher.find()) {
	            String code = matcher.group(1);        // ex: A1a
	            String description = matcher.group(2); // ex: Virage dangereux a droite
	            map.put(code, description);
	        }
	    });
	    
	    return map;
	}
	
	public static String[] folderCodes(String datasetPath) {
		/// récupère les classes "codés" à partir de la dataset
		
		File trainDir = new File(datasetPath);
        String[] folderCodes = trainDir.list((dir, name) -> new File(dir, name).isDirectory());
        java.util.Arrays.sort(folderCodes); 
        
        return folderCodes;
		
	}
	
	
	
	public static int bestClass(float[][] output) {
		//fonction qui retourne la meilleure classe à partir de la sortie (prédiction) du RNC
		int bestClass = 0;
        float maxProb = -1f;
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                bestClass = i;
            }
        }
        return bestClass;
	}
	
	public static float bestClassProba(float[][] output) {
		//fonction qui retourne la proba de la meilleure classe 
		int bestClass = 0;
        float maxProb = -1f;
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                bestClass = i;
            }
        }
        return maxProb;
	}
	
	public static String bestClassName(int bestClass, Map<String, String> indexToCode, Map<String, String> codeToName ) {
		//fonction qui renvoie le nom lisible de la meilleure classe
		
        String bestIndex = String.valueOf(bestClass);
        
         // Traduction :
         // Index 32 -> "B14_50" -> "Limitation de vitesse (50km/h)"
         String code = indexToCode.get(bestIndex);
         String className = codeToName.getOrDefault(code, code);
         
         return className;
	}
	
	public static float[]  bgr2rgb(float[] floatData, Mat resized) {
		int width = resized.cols();
		int height = resized.rows();
		int channels = resized.channels();

		byte[] data = new byte[width * height * channels];
		resized.get(0, 0, data);

		floatData = new float[width * height * channels];

		int idx = 0;

		for (int row = 0; row < height; row++) {
		    for (int col = 0; col < width; col++) {

		        int baseIndex = (row * width + col) * 3;

		        float b = (data[baseIndex] & 0xFF) / 255.0f;
		        float g = (data[baseIndex + 1] & 0xFF) / 255.0f;
		        float r = (data[baseIndex + 2] & 0xFF) / 255.0f;

		        floatData[idx++] = r;
		        floatData[idx++] = g;
		        floatData[idx++] = b;
		    }
		}
        
        return floatData;
	}
	
	public static void affichagePredictionRNC(Map<String, String> indexToCode, Map<String, String> codeToName, float[] floatData ,OrtEnvironment env,OrtSession session, Mat image) throws OrtException {
		long[] shape = {1, 64, 64, 3};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatData), shape)) {
            try (OrtSession.Result results = session.run(Collections.singletonMap("input", inputTensor))) {
                float[][] output = (float[][]) results.get(0).getValue();
                
                // meilleure class + proba
                int bestClass = bestClass(output);
                float maxProb = bestClassProba( output);
                String finalName = bestClassName(bestClass, indexToCode, codeToName);

                //System.out.println("Détection : " + finalName);

                // affichage
                String text = finalName + " (" + String.format("%.1f%%", maxProb * 100) + ")";
                Core.putText(image, text, new Point(30, 50), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0, 0), 2, Core.LINE_AA, false);
            }
        }
	}

}
