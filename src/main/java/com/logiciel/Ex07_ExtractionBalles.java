package com.logiciel;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Ex07_ExtractionBalles {

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

    // D�tection des zones rouges (comme dans le PDF)
    public static Mat DetecterRouge(Mat hsv) {
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

        Mat m = LectureImage("Images/Billard_Balls.jpg");
        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        ImShow("Image originale", m);

        // Conversion en HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_BGR2HSV);

        // Seuillage rouge
        Mat threshold = DetecterRouge(hsv);

        // D�tection des contours
        List<MatOfPoint> contours = new ArrayList<>();
        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Reconnaissance des cercles + extraction
        MatOfPoint2f contour2f = new MatOfPoint2f();
        float[] radius = new float[1];
        Point center = new Point();

        int compteur = 1;

        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            contour2f.fromList(contour.toList());
            Imgproc.minEnclosingCircle(contour2f, center, radius);

            double circleArea = Math.PI * radius[0] * radius[0];

            // V�rification : est-ce un cercle ?
            if (area / circleArea >= 0.8) {

                // Dessiner un cercle vert
                Core.circle(m, center, (int) radius[0], new Scalar(0, 255, 0), 2);

                // Dessiner un rectangle autour
                Rect rect = Imgproc.boundingRect(contour);
                Core.rectangle(m,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0), 2);

                // Extraction de la balle
                Mat balle = m.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width);

                ImShow("Balle " + compteur, balle);
                compteur++;
            }
        }

        ImShow("D�tection + Extraction", m);
    }
}
