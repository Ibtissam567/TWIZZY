package com.logiciel;

import org.opencv.core.*;

import org.opencv.highgui.Highgui;


public class Ex01_LectureAffichage {

    public static Mat lectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat m = lectureImage("Images/opencv.png");

        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        for (int i = 0; i < m.height(); i++) {
            for (int j = 0; j < m.width(); j++) {
                double[] BGR = m.get(i, j);

                if (BGR[0] == 255 && BGR[1] == 255 && BGR[2] == 255) {
                    System.out.print(".");
                } else {
                    System.out.print("+");
                }
            }
            System.out.println();
        }
    }
}
