package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Ex06_ReconnaissanceCercles {

    public static Mat LectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    public static void ImShow(String title, Mat img) {
        MatOfByte mob = new MatOfByte();
        Highgui.imencode(".png", img, mob);
        byte[] ba = mob.toArray();
        try {
            JFrame frame = new JFrame(title);
            frame.add(new JLabel(new ImageIcon(javax.imageio.ImageIO.read(new ByteArrayInputStream(ba)))));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) { e.printStackTrace(); }
    }

    // Détection des zones rouges (comme dans le PDF)
    public static Mat DetecterCerclesRouges(Mat hsv) {
        Mat t1 = new Mat();
        Mat t2 = new Mat();
        Mat threshold = new Mat();

        Core.inRange(hsv, new Scalar(0, 100, 100), new Scalar(10, 255, 255), t1);
        Core.inRange(hsv, new Scalar(160, 100, 100), new Scalar(179, 255, 255), t2);

        Core.bitwise_or(t1, t2, threshold);

        Imgproc.GaussianBlur(threshold, threshold, new Size(9, 9), 2, 2);

        return threshold;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat m = LectureImage("Images/circles_rectangles.jpg");
        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        ImShow("Image originale", m);

        // Conversion en HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_BGR2HSV);
        ImShow("HSV", hsv);

        // Seuillage rouge
        Mat threshold = DetecterCerclesRouges(hsv);
        ImShow("Seuillage", threshold);

        // Détection des contours
        List<MatOfPoint> contours = new ArrayList<>();
        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // 🔵 NOUVEAU : afficher juste les contours (comme dans le PDF)
        Mat drawingContours = Mat.zeros(threshold.size(), CvType.CV_8UC3);

        for (int i = 0; i < contours.size(); i++) {
            Scalar color = new Scalar(
                    Math.random() * 255,
                    Math.random() * 255,
                    Math.random() * 255
            );
            Imgproc.drawContours(drawingContours, contours, i, color, 2);
        }

        ImShow("Contours uniquement", drawingContours);

        // Reconnaissance des cercles
        MatOfPoint2f contour2f = new MatOfPoint2f();
        float[] radius = new float[1];
        Point center = new Point();

        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            contour2f.fromList(contour.toList());
            Imgproc.minEnclosingCircle(contour2f, center, radius);

            double circleArea = Math.PI * radius[0] * radius[0];

            // Vérification : est-ce un cercle ?
            if (area / circleArea >= 0.8) {
                Core.circle(m, center, (int) radius[0], new Scalar(0, 255, 0), 2);
            }
        }

        ImShow("Détection des cercles rouges", m);
    }
}
