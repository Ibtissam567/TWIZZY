package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class Ex10_ExtraireBallesRouges {
	public static Mat lectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // 1. Lecture de l'image

        Mat m = lectureImage("Images/Billard_Balls.jpg");       
        ImShow("Image originale", m);

        // 2. Conversion en HSV
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(m, hsvImage, Imgproc.COLOR_BGR2HSV);

        // 3. Seuillage pour le rouge (deux plages)
        Mat threshold1 = new Mat();
        Mat threshold2 = new Mat();
        Mat threshold = new Mat();

        // Rouge "bas" (0–10)
        Core.inRange(hsvImage, new Scalar(0, 100, 100), new Scalar(10, 255, 255), threshold1);
        // Rouge "haut" (160–179)
        Core.inRange(hsvImage, new Scalar(160, 100, 100), new Scalar(179, 255, 255), threshold2);

        Core.bitwise_or(threshold1, threshold2, threshold);

        // Optionnel : lissage
        Imgproc.GaussianBlur(threshold, threshold, new Size(9, 9), 2, 2);
        ImShow("Masque rouge", threshold);

        // 4. Détection des contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.Canny(threshold, threshold, 100, 200);
        Imgproc.findContours(threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // 5. Filtrer les contours qui sont des cercles
        Mat result = m.clone();
        MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
        float[] radius = new float[1];
        Point center = new Point();

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);

            double contourArea = Imgproc.contourArea(contour);
            if (contourArea < 50) continue; // petit bruit

            matOfPoint2f.fromList(contour.toList());
            Imgproc.minEnclosingCircle(matOfPoint2f, center, radius);

            double circleArea = Math.PI * radius[0] * radius[0];
            double ratio = contourArea / circleArea;

            // Si le contour "remplit" bien le cercle => c'est une balle
            if (ratio >= 0.8) {
                // Dessiner le cercle
            	Core.circle(result, center, (int) radius[0], new Scalar(0, 255, 0), 2);

                // 6. Extraire la balle (bounding box)
                Rect rect = Imgproc.boundingRect(contour);
                Mat tmp = m.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width);
                Mat ball = new Mat(tmp.size(), tmp.type());
                tmp.copyTo(ball);
                ImShow("Balle rouge", ball);
            }
        }

        ImShow("Balles rouges détectées", result);
    }

    // Petite fonction utilitaire pour afficher une Mat dans une fenêtre Swing
    public static void ImShow(String title, Mat img) {
        MatOfByte matOfByte = new MatOfByte();
        Highgui.imencode(".png", img, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        try {
            java.io.InputStream in = new java.io.ByteArrayInputStream(byteArray);
            java.awt.image.BufferedImage bufImage = javax.imageio.ImageIO.read(in);
            javax.swing.JFrame frame = new javax.swing.JFrame();
            frame.setTitle(title);
            frame.getContentPane().add(new javax.swing.JLabel(new javax.swing.ImageIcon(bufImage)));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
