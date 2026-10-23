package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.*;

public class Ex12_ExtractionCaracteristiques {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // --- Lecture de l'image ---
        Mat img = lectureImage("Images/Ball_13.png"); // choisis l’image que tu veux
        ImShow("Image originale", img);

        // --- 1. Conversion en niveaux de gris ---
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        // --- 2. Détection des points d'intérêt (ORB) ---
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        detector.detect(gray, keypoints);

        // --- 3. Extraction des descripteurs ---
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        Mat descriptors = new Mat();
        extractor.compute(gray, keypoints, descriptors);

        System.out.println("Nombre de keypoints détectés : " + keypoints.size());
        System.out.println("Taille des descripteurs : " + descriptors.size());

        // --- 4. Affichage des keypoints ---
        Mat output = new Mat();
        Features2d.drawKeypoints(img, keypoints, output, new Scalar(0, 255, 0), 0);

        ImShow("Points d'intérêt ORB", output);
    }

    // --- Lecture d'image ---
    public static Mat lectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    // --- Affichage ---
    public static void ImShow(String title, Mat img) {
        MatOfByte mob = new MatOfByte();
        Highgui.imencode(".png", img, mob);
        byte[] ba = mob.toArray();
        try {
            java.io.InputStream in = new java.io.ByteArrayInputStream(ba);
            java.awt.image.BufferedImage buf = javax.imageio.ImageIO.read(in);
            javax.swing.JFrame frame = new javax.swing.JFrame(title);
            frame.getContentPane().add(new javax.swing.JLabel(new javax.swing.ImageIcon(buf)));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
