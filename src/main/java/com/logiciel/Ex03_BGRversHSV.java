package com.logiciel;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.io.*;
import java.util.Vector;

public class Ex03_BGRversHSV {

    // Lecture d'une image (comme dans le PDF)
    public static Mat LectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    // Fonction ImShow du PDF
    public static void ImShow(String title, Mat img) {
        MatOfByte matOfByte = new MatOfByte();
        Highgui.imencode(".png", img, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            ImageIcon icon = new ImageIcon(javax.imageio.ImageIO.read(in));
            JFrame frame = new JFrame(title);
            frame.getContentPane().add(new JLabel(icon));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // MAIN
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Chemin correct dans ton projet
        Mat m = LectureImage("Images/hsv.png");

        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        // Conversion BGR → HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_BGR2HSV);

        // Affichage de l'image HSV
        ImShow("Image HSV", hsv);

        // Extraction des 3 canaux
        Vector<Mat> channels = new Vector<>();
        Core.split(hsv, channels);

        ImShow("Canal H (Hue)", channels.get(0));
        ImShow("Canal S (Saturation)", channels.get(1));
        ImShow("Canal V (Value)", channels.get(2));
    }
}
