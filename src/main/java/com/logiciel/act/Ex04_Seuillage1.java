package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Ex04_Seuillage1 {

    public static Mat LectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    public static void ImShow(String title, Mat img) {
        MatOfByte mob = new MatOfByte();
        Highgui.imencode(".png", img, mob);
        byte[] ba = mob.toArray();
        try {
            javax.swing.JFrame frame = new javax.swing.JFrame(title);
            frame.add(new javax.swing.JLabel(new javax.swing.ImageIcon(javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(ba)))));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) { e.printStackTrace(); }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat m = LectureImage("Images/circles.jpg");

        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        Mat hsv = new Mat();
        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_BGR2HSV);

        Mat threshold = new Mat();
        Core.inRange(hsv, new Scalar(0, 100, 100), new Scalar(10, 255, 255), threshold);

        // Lissage (comme dans le PDF)
        Imgproc.GaussianBlur(threshold, threshold, new Size(9, 9), 2, 2);

        ImShow("Cercles rouges (seuil unique)", threshold);
    }
}
