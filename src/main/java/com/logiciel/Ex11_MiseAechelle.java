package com.logiciel;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Ex11_MiseAechelle {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // --- Lecture des images ---
        // Choisis deux images de ton dossier Images
        Mat object = lectureImage("Images/Ball_13.png");
        Mat template = lectureImage("Images/Ball_three.png");

        ImShow("Objet original", object);
        ImShow("Template original", template);

        // --- 1. Mise à l'échelle de l'objet ---
        Mat scaledObject = new Mat();
        Imgproc.resize(object, scaledObject, template.size());

        ImShow("Objet redimensionné", scaledObject);

        // --- 2. Conversion en niveaux de gris ---
        Mat grayObject = new Mat();
        Mat grayTemplate = new Mat();

        Imgproc.cvtColor(scaledObject, grayObject, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(template, grayTemplate, Imgproc.COLOR_BGR2GRAY);

        // --- 3. Normalisation ---
        Core.normalize(grayObject, grayObject, 0, 255, Core.NORM_MINMAX);
        Core.normalize(grayTemplate, grayTemplate, 0, 255, Core.NORM_MINMAX);

        ImShow("Objet normalisé", grayObject);
        ImShow("Template normalisé", grayTemplate);
    }

    // --- Lecture d'image (comme tu l'as demandé) ---
    public static Mat lectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    // --- Affichage d'une Mat ---
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
