package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import javax.swing.*;
import java.io.*;
import java.util.Vector;

public class Ex02_CannauxCouleur {

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
        Mat m = LectureImage("Images/bgr.png");

        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        Vector<Mat> channels = new Vector<>();
        Core.split(m, channels); // B, G, R

        // Affichage des 3 canaux
        ImShow("Canal Bleu", channels.get(0));
        ImShow("Canal Vert", channels.get(1));
        ImShow("Canal Rouge", channels.get(2));
    }
}
